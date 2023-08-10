import type {
  ModelSpec,
  GenerateOptions,
  CommonOptions,
  LoadedModel,
  LoadReport,
  TemplateExpression,
  GenerationStreamHandler,
  TemplateExpressionOptions,
} from './types.js'

import { TargetDevice } from './types.js'
import doLoadModel from './loadModel.js'


let cachedModelAndSpec: { spec: ModelSpec, model: LoadedModel } | undefined;

// NOTE this currently only works for Llama 2 variations due to different wasm naming conventions
const guessModelSpecFromPrebuiltId = (id: string) => ({ // TODO generally works for currently known prebuilts
    modelWeightsConfigUrl: `https://huggingface.co/mlc-ai/mlc-chat-${id}/resolve/main/`,
    modelLibWasmUrl: `https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/${id}-webgpu.wasm`
})
/**
 * Load a model spec or try to guess it from a prebuilt-id.
 *
 * @remarks
 * Passing a prebuilt-id currently only works for Llama 2 models due to wasm naming differences.
 * Check here for some prebuilt-ids: {@link https://github.com/mlc-ai/binary-mlc-llm-libs}
 *
 *
 * @example
 * ```
 * const model = await loadModel('Llama-2-7b-chat-hf-q4f32_1')
 * ```
 */
export const loadModel = async (
  specOrId: ModelSpec | string,
  report?: (loadReport: LoadReport) => void,
  targetDevice: TargetDevice = TargetDevice.GPU,
): Promise<LoadedModel> => {
  const spec = typeof specOrId === 'string' ? guessModelSpecFromPrebuiltId(specOrId) : specOrId

  let loadReport: LoadReport = {
    modelSpec: spec,
    targetDevice
  }

  const updateReport = (update: LoadReport) => {
    loadReport = {
      ...loadReport,
      ...update
    }
    report?.(loadReport)
  }

  window.addEventListener('unhandledrejection', ev => {
    if (ev.reason instanceof Error) {
      if (ev.reason instanceof SyntaxError) { return } // bad JSON parse
      if (ev.reason.message.includes('Model cancelled')) { return }
    }

    updateReport({ error: ev.reason?.message ?? ev.reason })
  })

  if (cachedModelAndSpec?.spec.modelLibWasmUrl == spec.modelLibWasmUrl
      && cachedModelAndSpec?.spec.modelWeightsConfigUrl == cachedModelAndSpec?.spec.modelWeightsConfigUrl) {
    await cachedModelAndSpec.model.cancel()
    return cachedModelAndSpec.model
  }

  try {
    const model = await doLoadModel(spec, updateReport, targetDevice)

    cachedModelAndSpec = { model, spec }

    return model
  } catch (error) {
    if (error instanceof Error) {
      updateReport({ error: error.message })
    }

    throw error
  }
}

/// <reference types="vite/client" />
if (import.meta.hot) {
  import.meta.hot.accept()
}

const asOp = (expr: TemplateExpression, nextLiteral: string) => ({
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


const mergeAdModelGenConfig = (adConfig?: TemplateExpressionOptions, modelGenConfig?: GenerateOptions): CommonOptions => ({
  maxTokens: adConfig?.maxTokens ?? modelGenConfig?.maxTokens,
  temperature: adConfig?.temperature ?? modelGenConfig?.temperature,
  top_p: adConfig?.top_p ?? modelGenConfig?.top_p,
  validate: adConfig?.validate ?? modelGenConfig?.validate,
  sampler: adConfig?.sampler ?? modelGenConfig?.sampler
})

/**
 * A defined template ready for inferencing
 */
type Template = {
  /**
   * Collect the template as a string - optionally with a streaming handler
   */
  collect: (stream?: GenerationStreamHandler) => Promise<string>
  model: LoadedModel // TODO refactor to just cancel() -- this is only used to cancel the underlying model
}

/**
 * Template creation and helpers
 *
 * @example
 * ```
 * const createContext = ad(model)
 * const { template, a } = createContext('You are a Dungeon master', 'Create an interesting NPC')
 *
 * const npc = template`{
 *  "name": "${a('character name')}"
 * }`
 * ```
 */
type CreateTemplate = {
  template: (literals: TemplateStringsArray, ...expressions: TemplateExpression[]) => Template
  a: (prompt: string, accept?: TemplateExpressionOptions) => TemplateExpression
  __: (prompt: string, accept?: TemplateExpressionOptions) => TemplateExpression
}

/**
 * Establish a context for template inference
 *
 * @remarks
 * Set a common system prompt and preprompt, as well as common configuration ({@link TemplateExpressionOptions}) for child templates.
 */
type CreateTemplateContext = (system: string, preprompt?: string, config?: TemplateExpressionOptions) => CreateTemplate

/**
 * Create an ad-llama instance
 *
 * @example
 * ```
 * import { ad, loadModel } from 'ad-llama'
 * const createContext = ad(await loadModel('Llama-2-7b-chat-hf-q4f32_1'))
 * ```
 */
export const ad = (model: LoadedModel): CreateTemplateContext => {
  // TODO additional model configuration and context-local state goes here
  return (system: string, preprompt?: string, config?: TemplateExpressionOptions): CreateTemplate => ({
    template: (literals: TemplateStringsArray, ...expressions: TemplateExpression[]) => {
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
          await model.setContext(system, preprompt)

          if (stream) {
            stream({
              content: ops.reduce<string>( (completion, op) => {
                if (typeof(op) === 'string') {
                  return completion + op
                } else {
                  return completion + `\${'${op.prompt}'}`
                }
              }, head),
              type: 'template',
              system,
              preprompt
            })

            stream({
              content: head,
              type: 'lit'
            })
          }

          return ops.reduce<Promise<string>>(async (completion_, op) => {
            const completion = await completion_

            if (typeof(op) === 'string') {
              stream?.({
                content: op,
                type: 'lit'
              })
              return completion + op
            } else {
              return completion + await model.generate(
                op.prompt,
                completion,
                [op.stop, ...(op.accept?.stops ?? [])],
                {
                  stream,
                  ...mergeAdModelGenConfig(config, op.accept)
                }
              )
            }
          }, Promise.resolve(head))
        },
        model
      }
    },
    a: (prompt: string, accept?: TemplateExpressionOptions) => ({
      // TODO should I merge AdExprConfig from context here instead of within generate?
      prompt: `${config?.preword ?? 'Generate'} a ${prompt}`,
      options: accept,
    }),
    __: (prompt: string, accept?: TemplateExpressionOptions) => ({ prompt, options: accept, }),
  })
}

export { TargetDevice } from './types.js'

export * as validate from './validate.js'
export * as sample from './sample.js'
