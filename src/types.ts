import type { SamplerBuilder, Bias } from './sample.js'

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

export type CommonConfig = {
  maxTokens?: number
  temperature?: number
  top_p?: number
  validate?: {
    check?: (partial: string) => boolean
    transform?: (partial: string) => string
    retries: number
  }
  sampler?: SamplerBuilder
}

export type ModelGenConfig = {
  stream?: GenerationStreamHandler
} & CommonConfig

export type AdExprConfig = {
  preword?: string
  stops?: string[]
} & CommonConfig

export type AdExpr = {
  prompt: string
  accept?: AdExprConfig
}

export type AdTemplateExpression = AdExpr | string

export type LoadedModel = {
  setContext: (system: string, preprompt?: string) => Promise<void>
  generate: (
    prompt: string,
    completion: string,
    stops: string[],
    config?: ModelGenConfig
  ) => Promise<string>
  cancel: () => Promise<void>
  bias: Bias
} 

/**
 * Specifies where to retrieve model weights and configuration
 */
export type ModelSpec = {
  /**
   * URL of root of repo containing ndarray-cache.json and mlc-chat-config.json
   */
  modelWeightsConfigUrl: string // TODO ensure this ends in '/' or else the last section gets replaced by new URL()
  /**
   * URL of the compiled wasm for model
   */
  modelLibWasmUrl: string
}


