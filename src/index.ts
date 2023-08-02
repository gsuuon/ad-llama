import * as tvmjs from 'tvmjs'
import { Tokenizer } from '@mlc-ai/web-tokenizers'

type ModelSpec = {
  modelWeightsConfigUrl: string // url of root of repo containing ndarray-cache.json and mlc-chat-config.json
    // TODO ensure this ends in '/' or else the last section gets replaced by new URL()
  modelLibWasmUrl: string // url of the compiled wasm for model
}

const scope = (name?: string) => 'ad-llama' + name ? '/' + name : ''
const cacheScope = (name: string) => new tvmjs.ArtifactCache(scope(name))

// FIXME This only works for Llama-2 models because of the wasm name
export const guessModelSpecFromPrebuiltId = (id: string) => ({
    modelWeightsConfigUrl: `https://huggingface.co/mlc-ai/mlc-chat-${id}/resolve/main/`,
    modelLibWasmUrl: `https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/${id}-webgpu.wasm`
})

let loadedModel: { spec: ModelSpec, model: LoadedModel } | undefined;

/// <reference types="vite/client" />
if (import.meta.hot) {
  import.meta.hot.accept()
}

export const loadModel = async (spec: ModelSpec, targetDevice?: tvmjs.DLDevice): Promise<LoadedModel> => {
  if (loadedModel?.spec.modelLibWasmUrl == spec.modelLibWasmUrl && loadedModel?.spec.modelWeightsConfigUrl == loadedModel?.spec.modelWeightsConfigUrl) {
    return loadedModel.model
  }

  const gpu = await tvmjs.detectGPUDevice()
  if (gpu == undefined) { throw Error('Cannot find GPU in environment') }

  console.log('GPU found:\n', gpu)

  console.log('Loading model')
  console.log(spec)

  const configUrl = new URL('mlc-chat-config.json', spec.modelWeightsConfigUrl).href
  const configResponse = await cacheScope('config').fetchWithCache(configUrl)
  // TODO ArtifactCache error is probably too cryptic if configurl is invalid

  console.log('Loaded config')

  const wasm = await (
    spec.modelLibWasmUrl.includes('localhost') // never cache localhost
      ? fetch(spec.modelLibWasmUrl)
      : cacheScope('wasm').fetchWithCache(spec.modelLibWasmUrl)
  )

  const tvm = await tvmjs.instantiate(
    new Uint8Array(await wasm.arrayBuffer()),
    tvmjs.createPolyfillWASI()
  )

  if (targetDevice === undefined) {
    targetDevice = tvm.webgpu()
  }

  tvm.initWebGPU(gpu.device) // TODO Do I need to initWebGPU before fetchNDArrayCache? I'd prefer to defer this until later

  tvm.registerInitProgressCallback(report => console.log(report.text))

  console.log('Model weights download started')

  const loadingModelWeights = tvm.fetchNDArrayCache(spec.modelWeightsConfigUrl, targetDevice, scope('model'))

  const config = await configResponse.json()

  if (!Array.isArray(config.tokenizer_files)) {
    console.error(config)
    throw Error('Config json file is missing an array field named "tokenizer_files"')
  }

  const temperature = config.temperature ?? 1.0
  const top_p = config.top_p ?? 0.95

  const configTokenizerFiles = Object.entries({
    'tokenizer.model': Tokenizer.fromSentencePiece,
    'tokenizer.json': Tokenizer.fromJSON
  }).find(([file, _create]) => config.tokenizer_files.includes(file))

  if (configTokenizerFiles == undefined) {
    throw Error('Cant handle tokenizer files ' + config.tokenizer_files)
  }

  const [path, create] = configTokenizerFiles

  console.log('active tokenizer', path, create.name)

  const tokenizerResult =
    await cacheScope('model')
    .fetchWithCache(new URL(path, spec.modelWeightsConfigUrl).href)

  const tokenizer = await create(await tokenizerResult.arrayBuffer())
  console.log('Loaded tokenizer')

  // const __roundtripTokenizer = (text: string) => {
  //   console.log({roundtripIn: text})
  //   const out = tokenizer.decode(tokenizer.encode(text))
  //   console.log({roundtripOut: out})
  // }

  // __roundtripTokenizer('boopity boop a snoot')

  await loadingModelWeights
  console.log('Loaded weights')

  tvm.beginScope()

  const vm = tvm.detachFromCurrentScope(
    tvm.createVirtualMachine(targetDevice)
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

  await tvm.asyncLoadWebGPUPiplines(vm.getInternalModule())

  console.log('Loaded TVM pipelines')

  const tokenize = (text: string, prefix: number[] = [], postfix: number[] = []) => {
    // TODO figure out if we've exceeded max window size and handle
    const encodedText = tokenizer.encode(text)

    return [...prefix, ...encodedText, ...postfix]
  }

  let logitsOnCpu: tvmjs.NDArray | undefined;
  const sampleTokenFromLogits = async (logits: tvmjs.NDArray, temperature: number, top_p: number, mask?: number[]) => { // TODO mask
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

    await targetDevice?.sync()

    return tvm.sampleTopPFromLogits(logitsOnCpu, temperature, top_p)
  }

  const prefillStep = async (text: string) => {
    const tokens = tokenize(text)
    tvm.beginScope()

    const inputNdArray = tvm.empty([1, tokens.length], 'int32', targetDevice)
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

  const decodeStep = async (lastToken: number) => {
    tvm.beginScope()
    const inputNdArray = tvm.empty([1, 1], 'int32', targetDevice)
    inputNdArray.copyFrom([lastToken])

    const seqLenShape = tvm.makeShapeTuple([filledKvCacheLength + 1])
    const retValue = decode(inputNdArray, seqLenShape, kvCache, params)
    const logits = tvm.detachFromCurrentScope(retValue.get(0))

    tvm.endScope()

    filledKvCacheLength += 1

    return await sampleTokenFromLogits(logits, temperature, top_p)
  }

  let context = '<<sys>>You are a helpful assistant<</sys>>\n\n[INST]'

  const unfill = () => {
    clearKvCaches(kvCache)
    filledKvCacheLength = 0
    if (logitsOnCpu !== undefined) {
      logitsOnCpu.dispose()
      logitsOnCpu = undefined
    }
  }

  const model = {
    setContext: async (text: string) => {
      context = text
      console.log('Context:', context)

      // TODO prefill here, save kvCache, reset kvCache on each generate as necessary
      // Is that possible? can I prefill with existing kvCache?
    },
    generate: async (
      prompt: string,
      completion: string,
      stop: string,
      stream?: GenerationStreamHandler,
      maxTokens: number = 400
    ) => {
      const prefillText = `${context} Generate ${prompt} [/INST] ${completion}`
      console.info({generate: {prompt, stop, context: prefillText}})

      if (filledKvCacheLength > 0) {
        unfill()
      }

      const nextToken = await prefillStep(prefillText)

      if (nextToken === undefined) {
        throw Error('Prefilled with no sampled next token')
      }

      let tokens = [nextToken]
      let completedText = ''

      const getStopIndex = (text: string, tokenDecodedText: string, stop: string) => {
        // Check each new character in next token to see if it forms the stop sequence
        // with already completed text.
        // This gets around the issue where our stop is `"` but the next token generates `",` which
        // won't satisfy a straightforward endswith(stop)

        for (let i = tokenDecodedText.length; i >= 0; i--) {
          if (text.slice(0, text.length - i).endsWith(stop)) {
            return text.length - i - stop.length
          }
        }

        return -1
      }

      while (completedText.length < maxTokens) {
        const nextToken = await decodeStep(tokens[tokens.length - 1])

        tokens.push(nextToken)

        const updatedText = tokenizer.decode(new Int32Array(tokens))

        const tokenDecodedText = updatedText.slice(completedText.length)
          // decoding individual tokens and combining does not produce
          // same result as decoding seq of tokens

        const stopIdx = getStopIndex(updatedText, tokenDecodedText, stop)

        if (stopIdx !== -1) {
          const acceptedCompleteText = updatedText.slice(0, stopIdx)

          stream?.({
            content: acceptedCompleteText.slice(completedText.length),
            type: 'gen'
          })

          completedText = acceptedCompleteText
          break
        }

        stream?.({
          content: tokenDecodedText,
          type: 'gen'
        })

        completedText = updatedText
      }

      return completedText
    }
  }

  loadedModel = { spec, model }

  return model
}

type AdTemplateExpression = {
  prompt: string,
  accept: any // TODO
} | string

type GenerationStreamHandler = (partial: {content: string, type: 'gen' | 'lit'}) => void

const asOp = (expr: AdTemplateExpression, nextLiteral: string) => ({
  ...(typeof(expr) === 'string' ? {prompt: expr} : expr ),
  stop: nextLiteral.slice(0, 1) // determine stop from the next literal after expression
    // NOTE this isn't going to tokenize correctly necessarily
    // we will need to decode and then string compare
    // there are always multiple ways to encode the same bit of string depending on what's before and after
})

type Op = string | {
  prompt: string
  stop: string
  accept?: any
}

type LoadedModel = {
  setContext: (text: string) => Promise<void>
  generate: (
    prompt: string,
    completion: string,
    stop: string,
    stream?: GenerationStreamHandler
  ) => Promise<string>
}

// I think this would work better with a completion model than chat model
export const ad = (model: LoadedModel) => {
  return (system: string) => ({
    template: (literals: TemplateStringsArray, ...expressions: AdTemplateExpression[]) => {
      const [head, tail] = [literals[0], literals.slice(1)]

      let ops: Op[] = []

      // We make an assumption here that there is always one more literal than expression
      // Chrome seems to uphold this (template literal with only expression gets 2 empty strings)
      for (let i = 0; i < tail.length; i++) {
        ops.push(asOp(expressions[i], tail[i]))
        ops.push(tail[i])
      }

      return {
        collect: async (stream?: GenerationStreamHandler) => {
          await model.setContext(system)
          stream?.({
            content: head,
            type: 'lit'
          })

          return ops.reduce<Promise<string>>(async (completion_, op) => {
            const completion = await completion_

            if (typeof(op) === 'string') {
              stream?.({
                content: op,
                type: 'lit'
              })
              return completion + op
            } else {
              return completion + await model.generate(op.prompt, completion, op.stop, stream)
            }
          }, Promise.resolve(head))
        }
      }
    },
    a: (prompt: string, accept?: any) => ({
      prompt: `a ${prompt}`,
      accept,
    }),
  })
}
