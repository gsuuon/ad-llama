import { NDArray, Instance } from 'tvmjs'
import { Tokenizer } from '@mlc-ai/web-tokenizers'
import { AdExprConfig } from './types.js'

export type DeviceNDArray = {
  host: 'dev'
  data: NDArray
}

export type CpuNDArray = {
  host: 'cpu'
  data: NDArray
}

/**
 * Select relevant tokens -- called once for each sampling
 */
type SamplerSelector = (cpuLogits: CpuNDArray, tokens: number[], completion: string) => number[]
  // NOTE that since we return relevant tokens as an array, duplicates will have their bias applied twice (no change with gates)
  // I'm considering this a feature for now as a way to weigh relevant tokens, but consider changing number[] to a Set.

/**
 * Build the sample selector -- called once for each generation (each expression in the tagged template)
 */
type SamplerSelectBuilder = (priorCompletion: string) => SamplerSelector

/**
 * Build the sampler for the model -- called once per template
 */
type SamplerTemplateBuilder = (model: Model) => SamplerSelectBuilder

type Bias = (templateSampler: SamplerTemplateBuilder, weight: number) => SamplerBuilder
type Mask = (templateSampler: SamplerTemplateBuilder) => SamplerBuilder

export type Sampler = (
  cpuLogits: CpuNDArray,
  tokens: number[],
  completion: string,
  config: any
) => Promise<number>

type SamplerBuilder = (
  priorCompletion: string,
  temperature: number,
  top_p: number
) => Sampler

export type Model = {
  tvm: Instance
  tokenizer: Tokenizer
  sample: (logits: CpuNDArray, temperature: number, top_p: number) => Promise<number>
}

export type Biases = {
  prefer: Bias
  avoid: Bias
  // current implementation of avoid will likely not produce what we're looking for.
  // 1. there are always multiple token sequences which produce a string of characters, to alleviate we can:
  //   a. at selector, decode the current (entire) completion + relevant sequences
  //     * more performant
  //   b. at selector, decode entire voculary and check if it matches the provided xs (in case of oneOf)
  //     * more accurate
  //   c. at selector, decode relevant strings and whitespace prefix variants of those strings
  //     * most performant, least accurate
  // 2. avoid is case sensitive. not a problem with prefer, since we are driving towards specific tokens, not away from strings
  //   * if we have class: bias.avoid(oneOf(['ranger', 'cleric'])) we can still get 'Ranger' or 'Cleric'
  accept: Mask
  reject: Mask
}

type Config = {
  sampler: SamplerBuilder
}

export const buildBiases = (model: Model): Biases => {
  const { tvm, sample } = model

  const penalize = (buildModelSampler: SamplerTemplateBuilder, weight: number): SamplerBuilder => {
    const buildSelector = buildModelSampler(model)

    return (priorCompletion: string, temperature: number, top_p: number): Sampler => {
      const selector = buildSelector(priorCompletion)

      return async (cpuLogits, tokens, completion) => {
        const logits = cpuLogits.data

        const relevantTokens = selector(cpuLogits, tokens, completion)

        if (relevantTokens.length > 0) {
          console.log({penalizeTokens: relevantTokens, weight})
          tvm.beginScope()
          const relevantTokensNDArray = tvm.empty([1, relevantTokens.length], 'int32', tvm.cpu())
          relevantTokensNDArray.copyFrom(relevantTokens)

          tvm.applyRepetitionPenalty(logits, relevantTokensNDArray, weight)
          relevantTokensNDArray.dispose()
          tvm.endScope()
        }

        return await sample(cpuLogits, temperature, top_p)
      }
    }
  }

  // const mask = (
  //   selectBuilder: SelectBuilder,
  //   adjust: (relevantTokens: number[]) => (logit: number, idx: number) => number
  // ): SamplerBuilder => {
  //   const selector = selectBuilder(model)

  //   return async (cpuLogits, tokens, completion, config) => {
  //     const logits = cpuLogits.data

  //     const relevantTokens = selector(logits, tokens, completion)

  //     const start = performance.now()
  //     const modified = logits.toArray().map(adjust(relevantTokens))

  //     logits.copyFrom(new Float32Array(modified))

  //     console.log({ maskPerf: performance.now() - start })

  //     return tvm.sampleTopPFromLogits(logits, config.temperature, config.top_p)
  //   }
  // }

  return {
    prefer: (templateSampler, weight) => penalize(templateSampler, 1/weight),
    avoid: (templateSampler, weight) => penalize(templateSampler, weight),
    // accept: samplerInit => mask(
    //   samplerInit,
    //   (relevantTokens) => (logit, idx) => relevantTokens.includes(idx) ? logit : Number.NEGATIVE_INFINITY
    // ),
    // reject: selectBuilder => mask(
    //   samplerInit,
    //   (relevantTokens) => (logit, idx) => relevantTokens.includes(idx) ? Number.NEGATIVE_INFINITY : logit
    // )
  }
}

export const arrayStartsWith = <T>(starts: T[]) => (xs: T[]) => {
  for (let i = 0; i < starts.length; i++) {
    if (starts[i] !== xs[i]) {
      return false
    }
  }

  return true
}

/// Gets the tokens for extension given existing text. Can fail to produce tokens.
const encodeExtension = (tokenizer: Tokenizer, text: string, extension: string) => {
  const textTokens = tokenizer.encode(text)
  const lastToken = textTokens[textTokens.length - 1]

  for (let i = 0; i < text.length; i++) {
    const textTail = text.slice(text.length - i)

    const encoded = tokenizer.encode(textTail + extension)

    const lastTokenIdx = encoded.indexOf(lastToken)

    if (lastTokenIdx !== -1) {
      return encoded.slice(lastTokenIdx)
    }
  }
}

export const oneOf =
  (items: string[]): SamplerTemplateBuilder =>
    (model: Model): SamplerSelectBuilder =>
      (priorCompletion): SamplerSelector => {
        const encoded = items.map(
          item => {
            const extEncoding = encodeExtension(model.tokenizer, priorCompletion, item)

            if (extEncoding === undefined) {
              console.error('Failed to generate extension tokens, ignoring', item)
              return []
            }

            return Array.from(extEncoding)
          }
        )

        return (_logits, tokens, _completions) => {
          const filtered = encoded.filter(arrayStartsWith(tokens))
          const nextRelevantTokens = filtered.map(x => x[tokens.length] )
          console.log({
            nextRelevantTokens,
            nextRelevantChars: nextRelevantTokens.map(x => model.tokenizer.decode(new Int32Array([x]))),
            filtered,
            tokens: [...tokens],
            tokensChars: model.tokenizer.decode(new Int32Array(tokens)), // TODO fixme tokenizer.decode should take numbers
            items,
            encoded,
          })

          return nextRelevantTokens
        }
      }

export const chars = {
  number: oneOf(['0','1','2','3','4','5','6', '7', '8', '9',',','.'])
    // NOTE this encourages single char tokens which may result in less expected inferred text
}

export const fn = (model: Model): Config[] => {
  const bias = buildBiases(model)

  return [
    {
      sampler: bias.accept(chars.number)
    },
    {
      sampler: bias.prefer(oneOf(['boop', 'beep']), 100)
    }
  ]
}
