import * as tvmjs from 'tvmjs'
import { Tokenizer } from '@mlc-ai/web-tokenizers'

type ModelSpec = {
  modelWeightsConfigUrl: string // url of root of repo containing ndarray-cache.json and mlc-chat-config.json
    // TODO ensure this ends in '/' or else the last section gets replaced by new URL()
  modelLibWasmUrl: string // url of the compiled wasm for model
}

const scope = (name?: string) => 'ad-llama' + name ? '/' + name : ''
const cacheScope = (name: string) => new tvmjs.ArtifactCache(scope(name))

/// FIXME This only works for Llama-2 models because of the wasm name
const guessModelSpecFromPrebuiltId = (id: string) => ({
    modelWeightsConfigUrl: `https://huggingface.co/mlc-ai/mlc-chat-${id}/resolve/main/`,
    modelLibWasmUrl: `https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/${id}-webgpu.wasm`
})

const initialize = async () => {
  const gpu = await tvmjs.detectGPUDevice()
  if (gpu == undefined) { throw Error('Cannot find GPU in environment') }

  console.log('GPU found:\n', gpu)

  return {
    loadModel: async (spec: ModelSpec, targetDevice?: tvmjs.DLDevice): Promise<LoadedModel> => {
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

      const tokenizerResult =
        await cacheScope('model')
          .fetchWithCache(new URL(path, spec.modelWeightsConfigUrl).href)

      const tokenizer = await create(await tokenizerResult.arrayBuffer())
      console.log('Loaded tokenizer')

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
      const sampleTokenFromLogits = (logits: tvmjs.NDArray, temperature: number, top_p: number, mask?: number[]) => { // TODO mask
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

        return tvm.sampleTopPFromLogits(logitsOnCpu, temperature, top_p)
      }

      const prefillStep = (text: string) => {
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

        return sampleTokenFromLogits(logits, temperature, top_p)
      }

      const decodeStep = (lastToken: number) => {
        tvm.beginScope()
        const inputNdArray = tvm.empty([1, 1], 'int32', targetDevice)
        inputNdArray.copyFrom([lastToken])

        const seqLenShape = tvm.makeShapeTuple([filledKvCacheLength + 1])
        const retValue = decode(inputNdArray, seqLenShape, kvCache, params)
        const logits = tvm.detachFromCurrentScope(retValue.get(0))

        tvm.endScope()

        filledKvCacheLength += 1

        return sampleTokenFromLogits(logits, temperature, top_p)
      }

      let generatedTokens: number[] = []

      return {
        setContext: (text: string) => {
          // TODO this probably wont work as is since vm.prefill is likely to be stateful
          const nextToken = prefillStep(text)

          if (nextToken === undefined) {
            throw Error('Prefilled with no sampled next token')
          }

          generatedTokens.push(nextToken)

          console.log({nextToken})
        },
        generate: async (prompt: string, stop: string) => {
          // idea is we prefill with prompt as well as previously set context, but don't persist the prompt part

          // TODO this just generates based on setContext, ignoring prompt
          // FIXME this will probably not work at all beyond just one single generate

          if (generatedTokens.length === 0) {
            // TODO refactor so that this is not possible
            throw Error('Tried to generate without having called setContext')
          }

          let completeText = ''

          while (!completeText.endsWith(stop) && completeText.length < 100) {
            const nextToken = decodeStep(generatedTokens[generatedTokens.length - 1])

            if (nextToken === undefined) {
              throw Error('Prefilled with no sampled next token')
            }

            generatedTokens.push(nextToken)
            console.log({nextToken})
            // TODO hook for streaming generated tokens
            // stream(tokenizer.decode(new Int32Array([nextToken])))

            completeText = tokenizer.decode(new Int32Array(generatedTokens))
            console.log({completeText})
          }

          // we need to buffer at least the length of stop - if we see that the buffer is not stop, we output that section
          return completeText
        }
      }
    }
  }
}

type AdTemplateExpression = {
  prompt: string,
  accept: any // TODO
}


const asOp = (expr: AdTemplateExpression, nextLiteral: string) => ({
  ...expr,
  stop: nextLiteral.slice(0, 1) // determine stop from the next literal after expression
    // NOTE this isn't going to tokenize correctly necessarily
    // we will need to decode and then string compare
    // there are always multiple ways to encode the same bit of string depending on what's before and after
})

type Op = string | {
  prompt: string
  accept: any
  stop: string
}

type LoadedModel = {
  setContext: (text: string) => void
  generate: (prompt: string, stop: string) => Promise<string>
}

// I think this would work better with a completion model than chat model
const ad = (model: LoadedModel) => {
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
        collect: async () =>
          ops.reduce<Promise<string>>(async (completion_, op) => {
            const completion = await completion_

            if (typeof(op) === 'string') {
              return completion + op
            } else {
              model.setContext(system + completion)
              return completion + await model.generate(op.prompt, op.stop)
            }
          }, Promise.resolve(head))
      }
    },
    a: (prompt: string, accept?: any) => ({ prompt, accept }),
  })
}

export const test = async () => {
  const api = await initialize()
  const loadedModel = await api.loadModel(guessModelSpecFromPrebuiltId('Llama-2-7b-chat-hf-q4f32_1'))
  const generator = ad(loadedModel)
  const { template, a } = generator('You are a dungeon master. Generate a character based on the Dungeons and Dragons universe.')

  const result = template`
  {
    "description": "${a('description')}",
  }
  `

  console.log(await result.collect())
}
