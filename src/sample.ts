import { NDArray, Instance } from 'tvmjs'
import { Tokenizer } from '@mlc-ai/web-tokenizers'

export type DeviceNDArray = {
  host: 'dev'
  data: NDArray
}

export type CpuNDArray = {
  host: 'cpu'
  data: NDArray
}

/**
 * Select relevant tokens -- called once for each token generation
 */
type SamplerSelector = (cpuLogits: CpuNDArray, tokens: number[], completion: string) => number[]
  // NOTE that since we return relevant tokens as an array, duplicates will have their bias applied twice (no change with gates)
  // I'm considering this a feature for now as a way to weigh relevant tokens, but consider changing number[] to a Set.

/**
 * Build the sample selector -- called once for each generation (each expression in the tagged template)
 */
type SamplerSelectBuilder = (priorCompletion: string) => SamplerSelector

/**
 * Build the sampler given a model -- called once per template
 */
type SamplerTemplateBuilder = (model: Model) => SamplerSelectBuilder

type Bias = (templateSampler: SamplerTemplateBuilder, weight: number) => SamplerBuilder
type Mask = (templateSampler: SamplerTemplateBuilder) => SamplerBuilder

export type Sampler = (
  cpuLogits: CpuNDArray,
  tokens: number[],
  completion: string
) => number

export type SamplerBuilder = (
  priorCompletion: string,
  temperature: number,
  top_p: number
) => Sampler

export type Model = {
  tvm: Instance
  tokenizer: Tokenizer
  sample: (logits: CpuNDArray, temperature: number, top_p: number) => number
}

export type Biases = {
  prefer: Bias
  avoid: Bias
  accept: Mask
  reject: Mask
}

export const buildBiases = (model: Model): Biases => {
  const { tvm, sample } = model

  const penalize = (buildTemplateSampler: SamplerTemplateBuilder, weight: number): SamplerBuilder => {
    const buildSelector = buildTemplateSampler(model)

    return (priorCompletion: string, temperature: number, top_p: number): Sampler => {
      const selector = buildSelector(priorCompletion)

      return (cpuLogits, tokens, completion) => {
        const logits = cpuLogits.data

        const relevantTokens = selector(cpuLogits, tokens, completion)

        if (relevantTokens.length > 0) {
          console.log({
            penalizeTokens: relevantTokens,
            tokens,
            decodedTokens: model.tokenizer.decode(new Int32Array(tokens)),
            weight,
          })
          tvm.beginScope()
          const relevantTokensNDArray = tvm.empty([1, relevantTokens.length], 'int32', tvm.cpu())
          relevantTokensNDArray.copyFrom(relevantTokens)

          tvm.applyRepetitionPenalty(logits, relevantTokensNDArray, weight)
          relevantTokensNDArray.dispose()
          tvm.endScope()
        }

        return sample(cpuLogits, temperature, top_p)
      }
    }
  }

  const mask = (
    buildTemplateSampler: SamplerTemplateBuilder,
    adjust: (relevantTokens: number[]) => (logit: number, idx: number) => number
  ): SamplerBuilder => {
    const buildSelector = buildTemplateSampler(model)

    return (priorCompletion: string, temperature: number, top_p: number): Sampler => {
      const selector = buildSelector(priorCompletion)

      return (cpuLogits, tokens, completion) => {
        const logits = cpuLogits.data

        const relevantTokens = selector(cpuLogits, tokens, completion)

        if (relevantTokens.length > 0) {
          const start = performance.now()
          const modified = logits.toArray().map(adjust(relevantTokens))

          logits.copyFrom(new Float32Array(modified))

          console.log({ maskPerf: performance.now() - start })
        }

        return tvm.sampleTopPFromLogits(logits, temperature, top_p)
      }
    }
  }

  return {
    prefer: (templateSampler, weight) => penalize(templateSampler, 1/weight),
    avoid: (templateSampler, weight) => penalize(templateSampler, weight),
    accept: templateSampler => mask(
      templateSampler,
      (relevantTokens) => (logit, idx) => relevantTokens.includes(idx) ? logit : Number.NEGATIVE_INFINITY
    ),
    reject: templateSampler => mask(
      templateSampler,
      (relevantTokens) => (logit, idx) => relevantTokens.includes(idx) ? Number.NEGATIVE_INFINITY : logit
    )
  }
}

export const arrayStartsWith = <T>(starts: T[], xs: T[]) => {
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
      return encoded.slice(lastTokenIdx + 1)
    }
  }
}

export const oneOf = (items: string[]) => (model: Model) => (priorCompletion: string): SamplerSelector => {
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
    const filtered = encoded.filter(item => arrayStartsWith(tokens, item) && item.length > tokens.length)
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

export const consistsOf = (chars: string[], endings: string[]) => (model: Model) => (priorCompletion: string): SamplerSelector => {
  const encodedTokens = chars.map(
    char => {
      const extEncoding = encodeExtension(model.tokenizer, priorCompletion, char)
      // FIXME i probably want to do something different here for single characters
      // encodeExtension will produce the tokens which should continue from prior, but since we use the same tokens again
      // to bias, it's not going to work the same as one of. Instead we want to have single-char tokens that always
      // make sense given the existing sequence - i.e. if the first char is the first in sequence, then we want a sequence
      // start number char after that, we want single char number tokens that aren't BOS / subject to merge

      if (extEncoding === undefined) {
        console.error('Failed to generate extension tokens, ignoring', char)
        return []
      }

      if (endings.length > 0) {
        const endingsTokens = endings.map(ending => {
          const endingExtEncoding = encodeExtension(model.tokenizer, priorCompletion + char, ending)

          if (endingExtEncoding !== undefined) {
            return Array.from(endingExtEncoding)
          }

          return []
        })

        return Array.from(extEncoding).concat(endingsTokens.flat())
      }

      return Array.from(extEncoding)
    }
  ).flat()

  const encoded = Array.from(new Set(encodedTokens))

  return () => {
    console.log({
      consistsOfTokens: encoded,
      consistsOfChars: encoded.map(x => model.tokenizer.decode(new Int32Array([x])))
    })

    return encoded
  }
}

export const chars = {
  number: (endings: string[] = []) => consistsOf(['0','1','2','3','4','5','6', '7', '8', '9'], endings)
    // NOTE this encourages single char tokens which may result in less expected inferred text
    // I think there are no double-number tokens in llama 2 tokenizer vocab
}
