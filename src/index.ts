import type {
  ModelSpec,
  LoadedModel,
  LoadReport,
  AdTemplateExpression,
  GenerationStreamHandler,
} from './types.js'

import { TargetDevice } from './types.js'
import doLoadModel from './loadModel.js'

// FIXME This only works for Llama-2 models because of the wasm name
export const guessModelSpecFromPrebuiltId = (id: string) => ({
    modelWeightsConfigUrl: `https://huggingface.co/mlc-ai/mlc-chat-${id}/resolve/main/`,
    modelLibWasmUrl: `https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/${id}-webgpu.wasm`
})

let cachedModelAndSpec: { spec: ModelSpec, model: LoadedModel } | undefined;


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
    const reason = ev.reason?.message ?? ev.reason

    if (typeof(reason) === 'string') {
      if (reason.includes('Model cancelled')) { return }
      if (reason.includes('Bad control')) { return } // bad JSON parse
    }

    updateReport({ error: reason?.message ?? reason })
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

// I think this would work better with a completion model than chat model
export const ad = (model: LoadedModel) => {
  // TODO these are here to so that they're only available after loading a model
  // Reconsider if that design still makes sense. Maybe it'd be useful to define templates without
  // having a model yet.
  // The idea was that you could rely on intellisense alone to figure what to call to when getting started
  return (system: string, preprompt?: string) => ({
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
                stream,
                op.accept?.maxTokens
              )
            }
          }, Promise.resolve(head))
        },
        model
      }
    },
    a: (prompt: string, accept?: any) => ({
      prompt: `Generate a ${prompt}`,
      accept,
    }),
    __: (prompt: string, accept?: any) => ({ prompt, accept, }),
  })
}

export { TargetDevice } from './types.js'
