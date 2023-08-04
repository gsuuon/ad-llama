import type { NDArray } from 'tvmjs'
import { ArtifactCache, detectGPUDevice, instantiate, createPolyfillWASI } from 'tvmjs'
import { Tokenizer } from '@mlc-ai/web-tokenizers'

import { TargetDevice } from './types.js'

import type {
  ModelSpec,
  LoadedModel,
  LoadReport,
  ModelGenConfig,
} from './types.js'

enum ModelState {
  Waiting,
  Running,
  Cancelling,
}

const scope = (name?: string) => 'ad-llama' + name ? '/' + name : ''
const cacheScope = (name: string) => new ArtifactCache(scope(name))

export default async (
  spec: ModelSpec,
  updateReport: (loadReport: LoadReport) => void,
  targetDevice: TargetDevice,
): Promise<LoadedModel> => {

  const configUrl = new URL('mlc-chat-config.json', spec.modelWeightsConfigUrl).href
  const configResponse = await cacheScope('config').fetchWithCache(configUrl)
  // TODO ArtifactCache error is probably too cryptic if configurl is invalid

  updateReport({ loadModelConfig: 'done' })

  const wasm = await (
    spec.modelLibWasmUrl.includes('localhost') // never cache localhost
      ? fetch(spec.modelLibWasmUrl)
      : cacheScope('wasm').fetchWithCache(spec.modelLibWasmUrl)
  )

  const tvm = await instantiate(
    new Uint8Array(await wasm.arrayBuffer()),
    createPolyfillWASI()
  )

  const device = targetDevice === TargetDevice.GPU ? tvm.webgpu() : tvm.cpu()

  if (targetDevice === TargetDevice.GPU) {
    updateReport({detectGPU: 'waiting'})

    const gpu = await detectGPUDevice()
    if (gpu == undefined) {
      updateReport({detectGPU: 'failed'})
      throw Error('Cannot find GPU in environment')
    }

    updateReport({
      detectGPU: gpu.adapterInfo.vendor,
      loadModelConfig: 'waiting'
    })

    tvm.initWebGPU(gpu.device) 
  }

  let isLoadingGpuShaders = false

  tvm.registerInitProgressCallback(report => {
    if (isLoadingGpuShaders) {
      updateReport({ loadGPUShaders: report.progress })
    } else {
      if (report.cacheOnly) {
        updateReport({ loadModelFromCache: report.progress })
      } else {
        updateReport({ loadModelFromWeb: report.progress })
      }
    }
  })

  updateReport({ loadModel: 'waiting' })

  const config = await configResponse.json()

  if (!Array.isArray(config.tokenizer_files)) {
    console.error(config)

    const err = 'Config json file is missing an array field named "tokenizer_files"'
    updateReport({ loadModelConfig: err })
    throw Error(err)
  }

  const temperature_ = config.temperature ?? 1.0
  const top_p_ = config.top_p ?? 0.95

  updateReport({ loadTokenizer: 'waiting' })

  const configTokenizerFiles = Object.entries({
    'tokenizer.model': Tokenizer.fromSentencePiece,
    'tokenizer.json': Tokenizer.fromJSON
  }).find(([file, _create]) => config.tokenizer_files.includes(file))

  if (configTokenizerFiles == undefined) {
    const err = `Cant handle tokenizer files ${config.tokenizer_files}`;
    updateReport({ loadTokenizer: err });
    throw Error(err);
  }

  const [path, create] = configTokenizerFiles

  const tokenizerResult =
    await cacheScope('model')
    .fetchWithCache(new URL(path, spec.modelWeightsConfigUrl).href)

  const tokenizer = await create(await tokenizerResult.arrayBuffer())

  updateReport({ loadTokenizer: 'done' })

  await tvm.fetchNDArrayCache(spec.modelWeightsConfigUrl, device, scope('model'))

  updateReport({ loadModel: 'done' })

  tvm.beginScope()

  const vm = tvm.detachFromCurrentScope(
    tvm.createVirtualMachine(device)
  )

  const prefill = tvm.detachFromCurrentScope(
    vm.getFunction('prefill')
  )

  const decode = tvm.detachFromCurrentScope(
    vm.getFunction('decode')
  )

  const params = tvm.detachFromCurrentScope(
    tvm.getParamsFromCache('param', -1)
  )

  const createKvCache = vm.getFunction('create_kv_cache')

  const clearKvCaches = tvm.detachFromCurrentScope(
    tvm.getGlobalFunc('vm.builtin.attention_kv_cache_array_clear')
  )

  let kvCache = tvm.detachFromCurrentScope(createKvCache())
  let filledKvCacheLength = 0

  tvm.endScope()

  if (targetDevice === TargetDevice.GPU) {
    updateReport({ loadGPUShaders: 'waiting' })
    isLoadingGpuShaders = true

    await tvm.asyncLoadWebGPUPiplines(vm.getInternalModule())
    updateReport({ loadGPUShaders: 'done' })
  }

  const tokenize = (text: string, prefix: number[] = [], postfix: number[] = []) => {
    // TODO figure out if we've exceeded max window size and handle
    const encodedText = tokenizer.encode(text)

    return [...prefix, ...encodedText, ...postfix]
  }

  let logitsOnCpu: NDArray | undefined;

  const sampleTokenFromLogits = async (logits: NDArray, temperature?: number, top_p?: number, _mask?: number[]) => { // TODO mask
    tvm.beginScope()

    if (logitsOnCpu === undefined) {
      logitsOnCpu = tvm.detachFromCurrentScope(
        tvm.empty(logits.shape, logits.dtype, tvm.cpu())
      )
    } else {
      if (logits.shape[0] != logitsOnCpu.shape[0]) {
        throw Error('Logits changed shape')
      }
    }

    logitsOnCpu.copyFrom(logits)
    logits.dispose() // not sure if logits is at the right scope to be disposed from here

    // we skip a new begin/end scope here, check https://github.com/mlc-ai/web-llm/blob/824b7b3b2e22c69a2548f9516af9b9c7d012cd6b/src/llm_chat.ts#L317
    // if things break
    // TODO mask
    // we change the logits on cpu before doing sample

    tvm.endScope()

    await device?.sync()

    return tvm.sampleTopPFromLogits(logitsOnCpu, temperature ?? temperature_, top_p ?? top_p_)
  }

  const prefillStep = async (text: string, temperature?: number, top_p?: number) => {
    const tokens = tokenize(text)
    tvm.beginScope()

    const inputNdArray = tvm.empty([1, tokens.length], 'int32', device)
    inputNdArray.copyFrom(tokens)
    // this is not a direct translation, not sure if the nested begin/endscopes are necessary

    // TODO curpos param
    // https://github.com/mlc-ai/web-llm/blob/824b7b3b2e22c69a2548f9516af9b9c7d012cd6b/src/llm_chat.ts#L269
    const seqLenShape = tvm.makeShapeTuple([tokens.length]) 
    // NOTE llm_chat.ts conflates forward/decode here into a single forward function that handles both
    // i'm avoiding that, but if things break here check what's different

    const retValue = prefill(inputNdArray, seqLenShape, kvCache, params)
    // TODO is prefill stateful?

    const logits = tvm.detachFromCurrentScope(retValue.get(0))
    // skipping the endscope -> attachtocurrentscope because we're still in same scope here

    // TODO track seqlen, kv cache
    // it looks like filledkvcachelength is to set curpos for some steps (decode vs prefill)?
    tvm.endScope()

    // tvm.attachToCurrentScope(logits) // can I use unattached logits in sampleTokenFromLogits?

    filledKvCacheLength += tokens.length // not sure if this is supposed to only increment after decoding?

    return await sampleTokenFromLogits(logits, temperature, top_p)
  }

  const decodeStep = async (lastToken: number, temperature?: number, top_p?: number) => {
    tvm.beginScope()
    const inputNdArray = tvm.empty([1, 1], 'int32', device)
    inputNdArray.copyFrom([lastToken])

    const seqLenShape = tvm.makeShapeTuple([filledKvCacheLength + 1])
    const retValue = decode(inputNdArray, seqLenShape, kvCache, params)
    const logits = tvm.detachFromCurrentScope(retValue.get(0))

    tvm.endScope()

    filledKvCacheLength += 1

    return await sampleTokenFromLogits(logits, temperature, top_p)
  }

  let modelState: ModelState = ModelState.Waiting
  let system_ = '<<sys>>You are a helpful assistant<</sys>>\n\n'
  let preprompt_ = '[INST]';

  const unfill = () => {
    clearKvCaches(kvCache)
    filledKvCacheLength = 0
    if (logitsOnCpu !== undefined) {
      logitsOnCpu.dispose()
      logitsOnCpu = undefined
    }
  }

  const model = {
    setContext: async (system: string, preprompt?: string) => {
      system_ = `<<sys>>${system}<</sys>>\n\n`
      preprompt_ = preprompt ? `[INST] ${preprompt}` : preprompt_

      console.log('Context:', system, preprompt)

      // TODO prefill here, save kvCache, reset kvCache on each generate as necessary
      // Is that possible? can I prefill with existing kvCache?
    },
    generate: async (
      prompt: string,
      completion: string,
      stops: string[],
      config?: ModelGenConfig
    ) => {
      modelState = ModelState.Running as ModelState

      const temperature = config?.temperature
      const top_p = config?.top_p
      const maxTokens = config?.maxTokens ?? 400

      const prefillText = `${system_}${preprompt_} ${prompt} [/INST] ${completion}`
      console.info({generate: {prompt, stops, context: prefillText}})

      if (filledKvCacheLength > 0) {
        unfill()
      }

      const nextToken = await prefillStep(prefillText, temperature, top_p)

      if (nextToken === undefined) {
        throw Error('Prefilled with no sampled next token')
      }

      let tokens = [nextToken]
      let completedText = ''

      const getStopIndex = (text: string, tokenDecodedText: string, stops: string[]) => {
        // Check each new character in next token to see if it forms the stop sequence
        // with already completed text.
        // This gets around the issue where our stop is `"` but the next token generates `",` which
        // won't satisfy a straightforward endswith(stop)

        for (let i = tokenDecodedText.length; i >= 0; i--) {
          for (const stop of stops) {
            if (text.slice(0, text.length - i).endsWith(stop)) {
              return text.length - i - stop.length
            }
          }
        }

        return -1
      }

      // TODO eos token
      while (!(modelState === ModelState.Cancelling) && completedText.length < maxTokens) {
        const nextToken = await decodeStep(tokens[tokens.length - 1], temperature, top_p)

        tokens.push(nextToken)

        const updatedText = tokenizer.decode(new Int32Array(tokens))

        const tokenDecodedText = updatedText.slice(completedText.length)
          // decoding individual tokens and combining does not produce
          // same result as decoding seq of tokens

        const stopIdx = getStopIndex(updatedText, tokenDecodedText, stops)

        if (stopIdx !== -1) {
          const acceptedCompleteText = updatedText.slice(0, stopIdx)

          config?.stream?.({
            content: acceptedCompleteText.slice(completedText.length),
            type: 'gen',
            prompt
          })

          completedText = acceptedCompleteText
          break
        }

        config?.stream?.({
          content: tokenDecodedText,
          type: 'gen',
          prompt
        })

        completedText = updatedText
      }

      if (modelState === ModelState.Cancelling) {
        modelState = ModelState.Waiting
        unfill()
        throw Error('Model cancelled')
      }

      modelState = ModelState.Waiting

      return completedText
    },
    cancel: async () => {
      if (modelState === ModelState.Running) {
        modelState = ModelState.Cancelling
      }

      return new Promise<void>(resolve => setInterval(() => {
        if (modelState === ModelState.Waiting) {
          resolve()
        }
      }, 16))
    }
  }

  updateReport({ ready: true })

  return model
}


