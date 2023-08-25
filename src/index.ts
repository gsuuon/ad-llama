import type {
  ModelSpec,
  GenerateOptions,
  CommonOptions,
  LoadedModel,
  LoadReport,
  TemplateExpression,
  GenerationStreamHandler,
  TemplateExpressionOptions,
  TemplateContextOptions,
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

const keys = <T extends Object>(o: T) => Object.keys(o) as (keyof T)[]

export const report = (log: (text: string) => void) => {
  let lastReport: LoadReport | undefined;

  const _log = (label: string, value: any) => {
    const valueStr = value instanceof Object ? '\n' + JSON.stringify(value, null, 2) : value

    log(`${label}: ${valueStr}`)
  }

  return (report: LoadReport) => {
    if (lastReport !== undefined) {
      for (const label of keys(lastReport)) {
        if (lastReport[label] !== report[label]) {
          _log(label, report[label])
        }
      }
    } else {
      for (const label of keys(report)) {
        _log(label, report[label])
      }
    }

    lastReport = report
  }
}

/// <reference types="vite/client" />
if (import.meta.hot) {
  import.meta.hot.accept()
}

type Op = string | {
  prompt: string
  stop: string
  options?: TemplateExpressionOptions
} | {
  refExpr: WithRef<TemplateExpression>
  stop: string
}

const asOp = (
  expr: TemplateExpression | WithRef<TemplateExpression>,
  nextLiteral: string,
  configPreword: string
): Op | string => {
  const stop = nextLiteral.slice(0, 1)

  switch (typeof expr) {
    case 'function':
      return {
        refExpr: expr,
        stop
      }
    case 'string':
      return expr
    default:
      const preword = expr.preword ?? configPreword

      return {
        ...expr,
        prompt: preword === '' ? expr.prompt : preword + ' a ' + expr.prompt,
        stop
      }
  }
}

const expandIfRefOp = (op: Exclude<Op, string>, ref: (id: string) => string | undefined): {
  prompt: string,
  options?: TemplateExpressionOptions,
  stop: string
} => {
  if ('refExpr' in op) {
    const expr = op.refExpr(ref)

    if (typeof expr === 'string') {
      return {
        prompt: expr,
        stop: op.stop
      }
    } 

    return {
      ...expr,
      stop: op.stop
    }
  }

  return op
}

const mergeAdModelGenConfig = (exprOpts?: TemplateExpressionOptions, genOpts?: GenerateOptions): CommonOptions => ({
  maxTokens: exprOpts?.maxTokens ?? genOpts?.maxTokens,
  temperature: exprOpts?.temperature ?? genOpts?.temperature,
  top_p: exprOpts?.top_p ?? genOpts?.top_p,
  validate: exprOpts?.validate ?? genOpts?.validate,
  sampler: exprOpts?.sampler ?? genOpts?.sampler
})

/**
 * A defined template ready for inferencing
 */
export type Template = {
  /** Collect the template as a string - optionally with a streaming handler */
  collect: (stream?: GenerationStreamHandler) => Promise<string>
  /** Like collect but returns the completion and refs */
  collect_refs: (stream?: GenerationStreamHandler) => Promise<{completion: string, refs: Record<string, string>}>
  model: LoadedModel // TODO refactor to just cancel() -- this is only used to cancel the underlying model
}

/**
 * Template creator with context established
 *
 * @example
 * ```
 * const { context, a } = ad(model)
 * const dm = context('You are a Dungeon master', 'Create an interesting NPC')
 *
 * const npc = dm`{
 *  "name": "${a('character name')}"
 * }`
 * ```
 */
type CreateTemplateContext =
  (
    literals: TemplateStringsArray,
    ...expressions: (TemplateExpression|WithRef<TemplateExpression>)[]
  ) => Template

/**
 * Template context creator and template creation helpers
 * @example
 * ```ts
 * const { context, a, prompt } = ad(model)
 * const assistant = context('You are a helpful assistant')
 *
 * const template = assistant`{
 *  "petName": "${a('good name for a cat')}",
 *  "hairs": ${prompt('How many hairs does this cat have?')}
 * }`
 * ```
 */
type CreateTemplate = {
  /** Set a common system prompt and preprompt, as well as common configuration ({@link TemplateExpressionOptions}) for child templates. */
  context: (system: string, preprompt?: string, config?: TemplateContextOptions) => CreateTemplateContext
  /**
   * A template expression with the preword prepended to the prompt - defaults to 'Generate', which becomes 'Generate a'
   * @example
   * ```ts
   * const { context, a } = ad(model)
   * const assistant = context('You are a helpful assistant')
   *
   * const template = assistant`{
   *  "petName": "${a('good name for a cat')}"
   * }`
   * ```
   */
  a: (prompt: string, options?: TemplateExpressionOptions) => TemplateExpression
  /** A template expression with an unaltered prompt - the preword is ignored */
  prompt: (prompt: string, options?: TemplateExpressionOptions) => TemplateExpression
}

type WithRef<T> = (ref: (id: string) => string | undefined) => T

/**
 * Create an ad-llama instance
 *
 * @example
 * ```
 * import { ad, loadModel } from 'ad-llama'
 *
 * const { context, a, prompt } = ad(await loadModel('Llama-2-7b-chat-hf-q4f32_1'))
 * ```
 */
export const ad = (model: LoadedModel): CreateTemplate => {
  // TODO additional model configuration and context-local state goes here

  return {
    a: (prompt: string, options?: TemplateExpressionOptions): TemplateExpression => ({
      prompt,
      options,
    }),
    prompt: (prompt: string, options?: TemplateExpressionOptions): TemplateExpression => ({
      prompt,
      options,
      preword: '',
    }),
    context: (system: string, preprompt?: string, config?: TemplateContextOptions): CreateTemplateContext =>(literals, ...expressions) => {
      let refs: Record<string, string> = {}
      const ref = (id: string): string | undefined => refs[id]

      const [head, tail] = [literals[0], literals.slice(1)]

      let ops: Op[] = []

      // We make an assumption here that there is always one more literal than expression
      // Chrome seems to uphold this (template literal with only expression gets 2 empty strings)
      for (let i = 0; i < tail.length; i++) {
        ops.push(asOp(expressions[i], tail[i], config?.preword || 'Generate'))
        ops.push(tail[i])
      }

      const collect = async (stream?: GenerationStreamHandler) => {
        await model.setContext(system, preprompt)

        if (stream) {
          stream({
            content: ops.reduce<string>( (completion, op) => {
              if (typeof(op) === 'string') {
                return completion + op
              } else {
                if ('refExpr' in op) {
                  const expr = op.refExpr(x => `(ref: ${x})`)
                  console.log('template refExpr', {expr})
                  return completion + `\${'${typeof expr === 'string' ? expr : expr.prompt}'}`
                } else {
                  return completion + `\${'${op.prompt}'}`
                }
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
            const { options, prompt, stop } = expandIfRefOp(op, ref)

            const generated = await model.generate(
              prompt,
              completion,
              [stop, ...(options?.stops ?? [])],
              {
                stream,
                ...mergeAdModelGenConfig(config, options)
              }
            )

            if (options?.id !== undefined) {
              refs[options.id] = generated
            }

            return completion + generated
          }
        }, Promise.resolve(head))
      }

      return {
        collect,
        collect_refs: async (stream?: GenerationStreamHandler) => {
          const completion = await collect(stream)

          return {
            completion,
            refs
          }
        },
        model
      }
    },
  }
}

export { TargetDevice, StreamPartial, LoadedModel, GenerationStreamHandler } from './types.js'

export * as validate from './validate.js'
export * as sample from './sample.js'
