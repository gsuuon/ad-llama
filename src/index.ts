import type {
  ModelSpec,
  ModelGenConfig,
  CommonConfig,
  LoadedModel,
  LoadReport,
  AdTemplateExpression,
  GenerationStreamHandler,
  AdExprConfig,
  AdExpr
} from './types.js'

import { TargetDevice } from './types.js'
import doLoadModel from './loadModel.js'

/**
 * Guess a ModelSpec based on a given prebuilt id (one of the mlc-llm prebuilts)
 * NOTE this currently only works for Llama 2 variations due to different wasm naming conventions
 */
export const guessModelSpecFromPrebuiltId = (id: string) => ({ // TODO generally works for currently known prebuilts
    modelWeightsConfigUrl: `https://huggingface.co/mlc-ai/mlc-chat-${id}/resolve/main/`,
    modelLibWasmUrl: `https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/${id}-webgpu.wasm`
})

let cachedModelAndSpec: { spec: ModelSpec, model: LoadedModel } | undefined;

/**
 * Load a model to be used with ad
 *
 * @example
 * ```
 * import { loadModel, guessModelSpecFromPrebuiltId } from 'ad-llama'
 * const model = await loadModel(guessModelSpecFromPrebuiltId('Llama-2-7b-chat-hf-q4f32_1')
 * ```
 */
export const loadModel = async (
  spec: ModelSpec,
  report?: (loadReport: LoadReport) => void,
  targetDevice: TargetDevice = TargetDevice.GPU,
): Promise<LoadedModel> => {
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


const mergeAdModelGenConfig = (adConfig?: AdExprConfig, modelGenConfig?: ModelGenConfig): CommonConfig => ({
  maxTokens: adConfig?.maxTokens ?? modelGenConfig?.maxTokens,
  temperature: adConfig?.temperature ?? modelGenConfig?.temperature,
  top_p: adConfig?.top_p ?? modelGenConfig?.top_p,
  validate: adConfig?.validate ?? modelGenConfig?.validate,
  sampler: adConfig?.sampler ?? modelGenConfig?.sampler
})

/**
 * A defined template ready for inferencing
 */
export type AdTemplate = {
  /**
   * Collect the template as a string - optionally with a streaming handler
   */
  collect: (stream?: GenerationStreamHandler) => Promise<string>
  model: LoadedModel // TODO refactor to just cancel() -- this is only used to cancel the underlying model
}

/**
 * Methods for constructing a template
 *
 * @example
 * ```
 * const generate = ad(model)
 * const { template, a } = generate('You are a Dungeon master', 'Create an interesting NPC')
 *
 * const adTemplate = template`{
 *  "name": "${a('character name')}"
 * }`
 * ```
 */
export type AdTemplateBuilder = {
  template: (literals: TemplateStringsArray, ...expressions: AdTemplateExpression[]) => AdTemplate
  a: (prompt: string, accept?: AdExprConfig) => AdExpr
  __: (prompt: string, accept?: AdExprConfig) => AdExpr
}

/**
 * Establish context for template inference
 */
export type AdTemplateContext = (system: string, preprompt?: string, config?: AdExprConfig) => AdTemplateBuilder

/**
 * Create an ad-llama instance
 */
export const ad = (model: LoadedModel): AdTemplateContext => {
  // TODO additional model configuration and context-local state goes here
  return (system: string, preprompt?: string, config?: AdExprConfig): AdTemplateBuilder => ({
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
    a: (prompt: string, accept?: AdExprConfig) => ({
      prompt: `${config?.preword ?? 'Generate'} a ${prompt}`,
      accept,
    }),
    __: (prompt: string, accept?: AdExprConfig) => ({ prompt, accept, }),
  })
}

export { TargetDevice } from './types.js'

export * as validate from './validate.js'
export * as sample from './sample.js'
