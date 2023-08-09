import type { NDArray } from 'tvmjs'
import { ArtifactCache, detectGPUDevice, instantiate, createPolyfillWASI } from 'tvmjs'
import { Tokenizer } from '@mlc-ai/web-tokenizers'

import type { DeviceNDArray, CpuNDArray } from './sample.js'

import { TargetDevice } from './types.js'
import { buildBiases } from './sample.js'


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

  updateReport({ loadTokenizer: 'waiting' })

  const configTokenizerFiles = Object.entries({
    'tokenizer.model': Tokenizer.fromSentencePiece,
    'tokenizer.json': Tokenizer.fromJSON
  }).find(([file, _create]) => config.tokenizer_files.includes(file))
    // preference comes from the order of tokenizer_files -- seems like .json is preferred over .model

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
  const w = window as any
  w.encode = (x: string) => Array.from(tokenizer.encode(x))
  w.decode = (xs: number[]) => tokenizer.decode(new Int32Array(xs))

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

  const logitsOnCpuCopyFromAndDispose = (() => {
    let logitsOnCpu: NDArray | undefined;

    return async (ndarray: DeviceNDArray): Promise<CpuNDArray> => { // WTB linear types
      const logits = ndarray.data

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
      logits.dispose()

      tvm.endScope()
      await device?.sync()

      return {
        data: logitsOnCpu,
        host: 'cpu'
      }
    }
  })()

  const sampleTokenFromLogits = (ndarray: CpuNDArray, temperature: number, top_p: number) => {
    return tvm.sampleTopPFromLogits(ndarray.data, temperature, top_p)
  }

  const prefillStep = (text: string): DeviceNDArray => {
    const tokens = tokenize(text)
    tvm.beginScope()

    const inputNdArray = tvm.empty([1, tokens.length], 'int32', device)
    inputNdArray.copyFrom(tokens)

    const retValue = prefill(
      inputNdArray,
      tvm.makeShapeTuple([tokens.length]),
      kvCache,
      params
    )

    const logits = tvm.detachFromCurrentScope(retValue.get(0))
    tvm.endScope()

    filledKvCacheLength += tokens.length

    return {
      host: 'dev',
      data: logits,
    }
  }

  const decodeStep = (lastToken: number): DeviceNDArray => {
    tvm.beginScope()

    const inputNdArray = tvm.empty([1, 1], 'int32', device)
    inputNdArray.copyFrom([lastToken])

    const retValue = decode(
      inputNdArray,
      tvm.makeShapeTuple([filledKvCacheLength + 1]),
      kvCache,
      params
    )
    const logits = tvm.detachFromCurrentScope(retValue.get(0))

    tvm.endScope()

    filledKvCacheLength += 1

    return {
      data: logits,
      host: 'dev',
    }
  }

  let modelState: ModelState = ModelState.Waiting
  let system_ = '<<sys>>You are a helpful assistant<</sys>>\n\n'
  let preprompt_ = '[INST]';

  const unfill = () => {
    clearKvCaches(kvCache)
    filledKvCacheLength = 0
  }

  const generate = async (
    prompt: string,
    priorCompletion: string,
    stops: string[],
    config?: ModelGenConfig
  ): Promise<string> => {
    modelState = ModelState.Running as ModelState

    let tokens: number[] = []
    let completion = ''

    const temperature = config?.temperature ?? 1.0
    const top_p = config?.top_p ?? 0.95
    const maxTokens = config?.maxTokens ?? 400

    const buildSampler = config?.sampler
    const sample =
      buildSampler
      ? buildSampler(priorCompletion, temperature, top_p)
      : (logits: CpuNDArray) => sampleTokenFromLogits(logits, temperature, top_p)

    const prefillText = `${system_}${preprompt_} ${prompt} [/INST] ${priorCompletion}`
    console.info({generate: {prompt, stops, context: prefillText}})

    if (filledKvCacheLength > 0) {
      unfill()
    }

    const nextToken = sample(
      await logitsOnCpuCopyFromAndDispose(prefillStep(prefillText)),
      tokens,
      completion
    )

    if (nextToken === undefined) {
      throw Error('Prefilled with no sampled next token')
    }

    tokens.push(nextToken)

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
    while (!(modelState === ModelState.Cancelling) && completion.length < maxTokens) {
      const nextToken = sample(
        await logitsOnCpuCopyFromAndDispose(decodeStep(tokens[tokens.length - 1])),
        tokens,
        completion
      )

      tokens.push(nextToken)

      const updatedText = tokenizer.decode(new Int32Array(tokens))

      const tokenDecodedText = updatedText.slice(completion.length)
      // decoding individual tokens and combining does not produce
      // same result as decoding seq of tokens

      const stopIdx = getStopIndex(updatedText, tokenDecodedText, stops)

      if (stopIdx !== -1) {
        const acceptedCompleteText = updatedText.slice(0, stopIdx)

        config?.stream?.({
          content: acceptedCompleteText.slice(completion.length),
          type: 'gen',
          prompt
        })

        completion = acceptedCompleteText
        break
      }

      config?.stream?.({
        content: tokenDecodedText,
        type: 'gen',
        prompt
      })

      completion = updatedText
    }

    if (modelState === ModelState.Cancelling) {
      modelState = ModelState.Waiting
      unfill()
      throw Error('Model cancelled')
    }

    modelState = ModelState.Waiting

    if (config?.validate) {
      if (config.validate.retries > 0 && config.validate.check && !config.validate.check(completion)) {
        config?.stream?.({
          type: 'ungen',
          tokenCount: tokens.length,
          content: completion
        })

        console.log({failedValidation: completion})

        return await generate(prompt, priorCompletion, stops, {
          ...config,
          validate: {
            ...config.validate,
            retries: config.validate.retries - 1,
          }
        })
      }

      if (config.validate.transform) {
        // We transform even if validation fails due to exhausting retries. Should we only transform if validate succeeds?
        config?.stream?.({
          type: 'ungen',
          tokenCount: tokens.length,
          content: completion
        })

        const transformed = config.validate.transform(completion)

        config?.stream?.({
          type: 'gen',
          content: transformed,
          prompt
        })

        return transformed
      }
    }

    return completion
  }

  const biases = buildBiases({ tvm, tokenizer, sample: sampleTokenFromLogits })

  const loadedModel = {
    generate,
    biases,
    setContext: async (system: string, preprompt?: string) => {
      system_ = `<<sys>>${system}<</sys>>\n\n`
      preprompt_ = preprompt ? `[INST] ${preprompt}` : preprompt_

      console.log('Context:', system, preprompt)

      // TODO prefill here, save kvCache, reset kvCache on each generate as necessary
      // Is that possible? can I prefill with existing kvCache?
      // This only saves prefilling the system + preprompt anyway - it won't do anything for generates since the generate prompt
      // goes before the completion body
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

  return loadedModel
}


