import type { Sampler } from './sample.js'

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

export type AdTemplateExpression = {
  prompt: string,
  accept: any // TODO
} | string

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
    check: (partial: string) => boolean
    transform: (partial: string) => string
    retries: number
  }
  sampler?: Sampler
}

export type ModelGenConfig = {
  stream?: GenerationStreamHandler
} & CommonConfig

export type AdConfig = {
  preword?: string
} & CommonConfig

export type LoadedModel = {
  setContext: (system: string, preprompt?: string) => Promise<void>
  generate: (
    prompt: string,
    completion: string,
    stops: string[],
    config?: ModelGenConfig
  ) => Promise<string>
  cancel: () => Promise<void>
}

export type ModelSpec = {
  modelWeightsConfigUrl: string // url of root of repo containing ndarray-cache.json and mlc-chat-config.json
    // TODO ensure this ends in '/' or else the last section gets replaced by new URL()
  modelLibWasmUrl: string // url of the compiled wasm for model
}


