import * as tvmjs from 'tvmjs'
import { Tokenizer } from '@mlc-ai/web-tokenizers'

type ModelSpec = {
  modelWeightsConfigUrl: string // url of root of repo containing ndarray-cache.json and mlc-chat-config.json
  modelLibWasmUrl: string // url of the compiled wasm for model
}

const scope = (name?: string) => 'ad-llama' + name ? '/' + name : ''

const guessModelSpecFromPrebuiltId = (id: string) => ({
    modelWeightsConfigUrl: `https://huggingface.co/mlc-ai/mlc-chat-${id}/resolve/main`,
    modelLibWasmUrl: `https://github.com/mlc-ai/binary-mlc-llm-libs/blob/main/${id}-webgpu.wasm`
})

const initialize = async () => {
  const gpu = await tvmjs.detectGPUDevice()
  if (gpu == undefined) { throw Error('Cannot find GPU in environment') }

  console.log('GPU found:\n', gpu)

  return {
    loadModel: async (spec: ModelSpec): Promise<LoadedModel> => {
      const configResponse = await fetch(new URL('mlc-chat-config.json', spec.modelWeightsConfigUrl))

      if (!configResponse.ok) {
        throw Error('Invalid model config url. Check that <url>/mlc-chat-config.json exists and is reachable.')
      }

      console.log('Loaded config')

      const wasm = await (
        spec.modelLibWasmUrl.includes('localhost')
          ? fetch(spec.modelLibWasmUrl)
          : new tvmjs.ArtifactCache(scope('wasm')).fetchWithCache(spec.modelLibWasmUrl)
      )

      const tvm = await tvmjs.instantiate(
        new Uint8Array(await wasm.arrayBuffer()),
        tvmjs.createPolyfillWASI()
      )

      tvm.initWebGPU(gpu.device) // TODO Do I need to initWebGPU before fetchNDArrayCache? I'd prefer to defer this until later

      tvm.registerInitProgressCallback( report => {
        console.log('Model fetch progress: ', report.progress)
      })

      console.log('Model weights download started')

      const loadingModelWeights = tvm.fetchNDArrayCache(spec.modelWeightsConfigUrl, tvm.webgpu(), scope('model'))

      const config = configResponse.json() as any
      const configTokenizerFiles = Object.entries({
        'tokenizer.model': Tokenizer.fromSentencePiece,
        'tokenizer.json': Tokenizer.fromJSON
      }).find(([file, _create]) => config.tokenizer_files.includes(file))

      if (configTokenizerFiles == undefined) {
        throw Error('Cant handle tokenizer files ' + config.tokenizer_files)
      }

      const [path, create] = configTokenizerFiles

      const tokenizerResult =
        await new tvmjs.ArtifactCache(scope('model'))
          .fetchWithCache(new URL(path, spec.modelWeightsConfigUrl).href)

      const tokenizer = create(await tokenizerResult.arrayBuffer())
      console.log('Loaded tokenizer')

      await loadingModelWeights
      console.log('Loaded weights')

      tvm.beginScope()

      const vm = tvm.detachFromCurrentScope(
        tvm.createVirtualMachine(tvm.webgpu())
      )

      const prefill = tvm.detachFromCurrentScope(
        vm.getFunction('prefill')
      )

      const decoding = tvm.detachFromCurrentScope(
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

      return {
        setContext: (text: string) => {},
        generate: async (prompt: string, stop: string) => {
          return prompt + 'TODO'
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
  stop: nextLiteral.slice(0, 1)
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

      // determine stops from the next literal after expression
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

const api = await initialize()
const loadedModel = await api.loadModel(guessModelSpecFromPrebuiltId('Llama-2-7b-chat-hf-q4f32_1'))
const generator = ad(loadedModel)
const { template, a } = generator('You are a dungeon master. Generate a character based on the Dungeons and Dragons universe.')

const result = template`{
  "name": "${a('name')}",
}`

console.log(await result.collect())
