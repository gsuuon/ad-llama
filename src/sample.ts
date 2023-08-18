import { NDArray, Instance } from 'tvmjs'
import { Tokenizer } from '@mlc-ai/web-tokenizers'

/** @internal */
export type DeviceNDArray = {
  host: 'dev'
  data: NDArray
}

/** @internal */
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
type CreateSamplerSelector = (priorCompletion: string, stops: string[]) => SamplerSelector

/**
 * Build the sampler given a model -- called once per template
 */
type CreateSamplerTemplate = (model: Model) => CreateSamplerSelector


/** @internal */
export type Sampler = (
  cpuLogits: CpuNDArray,
  tokens: number[],
  completion: string
) => number

/** @internal */
export type CreateSampler = (
  priorCompletion: string,
  stops: string[],
  temperature: number,
  top_p: number
) => Sampler

/** @internal */
export type Model = {
  tvm: Instance
  tokenizer: Tokenizer
  sample: (logits: CpuNDArray, temperature: number, top_p: number) => number
}

/** @internal */
export type Bias = {
  prefer: (templateSampler: CreateSamplerTemplate, weight: number) => CreateSampler
  avoid: (templateSampler: CreateSamplerTemplate, weight: number) => CreateSampler
  accept: (templateSampler: CreateSamplerTemplate) => CreateSampler
  reject: (templateSampler: CreateSamplerTemplate) => CreateSampler
}

/** @internal */
export const buildBias = (model: Model): Bias => {
  const { tvm, sample } = model

  const penalize = (buildTemplateSampler: CreateSamplerTemplate, weight: number): CreateSampler => {
    const buildSelector = buildTemplateSampler(model)

    return (priorCompletion, stops, temperature, top_p): Sampler => {
      const selector = buildSelector(priorCompletion, stops)

      return (cpuLogits, tokens, completion) => {
        const logits = cpuLogits.data

        const relevantTokens = selector(cpuLogits, tokens, completion)

        if (relevantTokens.length > 0) {
          console.debug('penalize', {
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
    buildTemplateSampler: CreateSamplerTemplate,
    adjust: (relevantTokens: number[]) => (logit: number, idx: number) => number
  ): CreateSampler => {
    const buildSelector = buildTemplateSampler(model)

    return (priorCompletion, stops, temperature, top_p): Sampler => {
      const selector = buildSelector(priorCompletion, stops)

      return (cpuLogits, tokens, completion) => {
        const logits = cpuLogits.data

        const relevantTokens = selector(cpuLogits, tokens, completion)

        if (relevantTokens.length > 0) {
          const modified = logits.toArray().map(adjust(relevantTokens))

          logits.copyFrom(new Float32Array(modified))
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

const arrayStartsWith = <T>(starts: T[], xs: T[]) => {
  for (let i = 0; i < starts.length; i++) {
    if (starts[i] !== xs[i]) {
      return false
    }
  }

  return true
}

/** Gets the tokens for extension given existing text. Can fail to produce tokens. */
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

/**
 * A template sampler creator where each sample has logits modified depending on the tokens which are still relevant given the already sampled tokens. For use with {@link LoadedModel}
 *
 * @remarks
 * For example, if we have `oneOf(['ranger', 'wizard'])` and we've already sampled 'w', the only next relevant tokens would be from 'izard'. If you want `oneOf` to stop at one of the choices, include the stop character (by default the next character after the expression), eg: `oneOf(['ranger"', 'wizard"'])`.
 *
 * @see {@link Bias}
 *
 * @example
 * ```
 * template`{
 *   "weapon": "${a('special weapon', {
 *     sampler: bias.prefer(oneOf(['Nun-chucks', 'Beam Cannon']), 10),
 *   })}"
 * }`
 * ```
 */
export const oneOf: (items: string[]) => CreateSamplerTemplate = items => model => priorCompletion => {
  const encoded = items.map(
    item => {
      const extEncoding = encodeExtension(model.tokenizer, priorCompletion, item)

      if (extEncoding === undefined) {
        console.warn(`Failed to generate extension token for \`${item}\` using simple encoding`)
        return Array.from(model.tokenizer.encode(item))
      }

      return Array.from(extEncoding)
    }
  )

  return (_logits, tokens, _completions) => {
    const filtered = encoded.filter(item => arrayStartsWith(tokens, item) && item.length > tokens.length)
    const nextRelevantTokens = filtered.map(x => x[tokens.length])

    console.debug('oneOf', {
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

/**
 * A template sampler creator for classes of characters.
 *
 * Each sample will have the same token logits modified based on the given relevant tokens. So in `consistsOf(['a','b'])`, every sample will have tokens for 'a' and 'b' modified.
 */
export const consistsOf = (chars: string[]) => (model: Model) => (priorCompletion: string, endings: string[]): SamplerSelector => {
  const encodedTokens = chars.map(
    char => {
      const extEncoding = encodeExtension(model.tokenizer, priorCompletion, char)
      // FIXME i probably want to do something different here for single characters
      // encodeExtension will produce the tokens which should continue from prior, but since we use the same tokens again
      // to bias, it's not going to work the same as one of. Instead we want to have single-char tokens that always
      // make sense given the existing sequence - i.e. if the first char is the first in sequence, then we want a sequence
      // start number char after that, we want single char number tokens that aren't BOS / subject to merge

      if (extEncoding === undefined) {
        console.warn(`Failed to generate extension token for \`${char}\` using simple encoding`)
        return Array.from(model.tokenizer.encode(char))
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

  return (_, tokens, completion) => {
    console.debug('consistsOf', {
      tokens: [...tokens],
      tokensChars: model.tokenizer.decode(new Int32Array(tokens)),
      completion,
      consistsOfTokens: encoded,
      consistsOfChars: encoded.map(x => model.tokenizer.decode(new Int32Array([x])))
    })

    return encoded
  }
}

export const chars = {
  number: consistsOf(['0','1','2','3','4','5','6', '7', '8', '9'])
    // NOTE this encourages single char tokens which may result in less expected inferred text
    // I think there are no double-number tokens in llama 2 tokenizer vocab
}
