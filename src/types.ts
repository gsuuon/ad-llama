import type { CreateSampler, Bias } from './sample.js'

export enum TargetDevice {
  CPU = 'cpu',
  GPU = 'gpu'
}

export type LoadReport = {
  loadModel?: 'waiting' | 'done'
  loadTokenizer?: 'waiting' | string | 'done'
  loadModelConfig?: 'waiting' | string | 'done'
  loadModelFromCache?: number
  loadModelFromWeb?: number
  loadGPUShaders?: 'waiting' | number | 'done'
  detectGPU?: 'waiting' | string | 'failed'
  targetDevice?: TargetDevice
  modelSpec?: ModelSpec
  ready?: boolean
  error?: any
}

export type StreamPartial = {
  type: 'lit'
  content: string
} | {
  type: 'gen'
  content: string
  prompt: string
} | {
  type: 'template'
  content: string
  system: string
  preprompt?: string
} | {
  type: 'ungen'
  content: string
  tokenCount: number
}

export type GenerationStreamHandler = (partial: StreamPartial) => void

export type CommonOptions = {
  maxTokens?: number
  temperature?: number
  top_p?: number
  validate?: {
    check?: (partial: string) => boolean
    transform?: (partial: string) => string
    retries?: number
  }
  sampler?: CreateSampler
}

export type GenerateOptions = {
  stream?: GenerationStreamHandler
} & CommonOptions

export type TemplateExpressionOptions = {
  stops?: string[]
  id?: string
} & CommonOptions

export type TemplateContextOptions =
  Omit<TemplateExpressionOptions, 'id'> & {
    preword?: string
  }

export type TemplateExpression = {
  prompt: string
  preword?: string | null
  options?: TemplateExpressionOptions
} | string

/**
 * The loaded model to give to {@link ad} to create an ad context.
 *
 * @example
 * ```
 * const { context, a, prompt } = ad(loadedModel)
 * ```
 */
export type LoadedModel = {
  generate: (params: {
    prompt: string,
    priorCompletion: string,
    stops: string[],
    system?: string,
    preprompt?: string
  }, config?: GenerateOptions) => Promise<string>
  cancel: () => Promise<void>
  bias: Bias
  encode: (text: string) => number[]
  decode: (tokens: number[]) => string
  totalTokenCount: number
}

/**
 * Specifies where to retrieve model weights and configuration
 */
export type ModelSpec = {
  /**
   * URL of root of repo containing ndarray-cache.json and mlc-chat-config.json For example:
   *
   * https://huggingface.co/mlc-ai/mlc-chat-Llama-2-7b-chat-hf-q4f32_1/resolve/main/
   *
   * @remarks
   * For hugging face, this root currently returns 404 -- that's fine, as long as the files themselves can be fetched.
   * Do not provide /tree/main URL's.
   */
  modelWeightsConfigUrl: string // TODO ensure this ends in '/' or else the last section gets replaced by new URL()
  /**
   * URL of the compiled wasm for model. For example:
   *
   * https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Llama-2-7b-chat-hf-q4f32_1-webgpu.wasm
   */
  modelLibWasmUrl: string
  /**
   * Size of the context window, overrides the number given in config metadata
   */
  contextWindowSize?: number
}


