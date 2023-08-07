import { NDArray, Instance } from 'tvmjs'
import type { Tokenizer } from '@mlc-ai/web-tokenizers'

type Selector = (logits: NDArray, tokens: number[], completion: string) => number[]

type Bias = (selector: SelectBuilder, weight: number) => Sampler
type Mask = (selector: SelectBuilder) => Sampler

type Sampler = (logits: NDArray, tokens: number[], completion: string, config: any) => number

export type Model = {
  tvm: Instance
  tokenizer: Tokenizer
}

type SelectBuilder = (model: Model) => Selector

export type Biases = {
  prefer: Bias
  avoid: Bias
  accept: Mask
  reject: Mask
}

type Config = {
  sampler: Sampler
}


export const buildBiases = (model: Model): Biases => {
  const { tvm } = model

  const penalize = (selectBuilder: SelectBuilder, weight: number) => {
    const selector = selectBuilder(model)

    return (logits: NDArray, tokens: number[], completion: string, config: any) => {
      const relevantTokens = selector(logits, tokens, completion)
      const arr = tvm.empty([relevantTokens.length, 1])
      arr.copyFrom(relevantTokens)

      tvm.applyRepetitionPenalty(logits, arr, weight)
      return tvm.sampleTopPFromLogits(logits, config.temperature, config.top_p)
    }
  }

  const mask = (selectBuilder: SelectBuilder, adjust: (relevantTokens: number[]) => (logit: number, idx: number) => number) => {
    const selector = selectBuilder(model)

    return (logits: NDArray, tokens: number[], completion: string, config: any) => {
      const relevantTokens = selector(logits, tokens, completion)

      const start = performance.now()
      const modified = logits.toArray().map(adjust(relevantTokens))

      logits.copyFrom(new Float32Array(modified))

      console.log({ maskPerf: performance.now() - start })

      return tvm.sampleTopPFromLogits(logits, config.temperature, config.top_p)
    }
  }

  return {
    prefer: (selectBuilder, weight) => penalize(selectBuilder, 1/weight),
    avoid: (selectBuilder, weight) => penalize(selectBuilder, weight),
    accept: selectBuilder => mask(
      selectBuilder,
      (relevantTokens) => (logit, idx) => relevantTokens.includes(idx) ? logit : Number.NEGATIVE_INFINITY
    ),
    reject: selectBuilder => mask(
      selectBuilder,
      (relevantTokens) => (logit, idx) => relevantTokens.includes(idx) ? Number.NEGATIVE_INFINITY : logit
    )
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

export const oneOf = (items: string[]): SelectBuilder => {
  return (model: Model) => {
    const encoded: number[][] = items.map(x => Array.from(model.tokenizer.encode(x)))

    return (_logits, tokens, _completions) => {
      return encoded.filter(arrayStartsWith(tokens)).map(x => x[tokens.length] )
    }
  }
}

const SCRATCH_Prebuilts = {
  number: oneOf(['0','1','2','3','4','5','6', '7', '8', '9',',','.'])
    // NOTE this encourages single char tokens which may result in less expected inferred text
}

export const fn = (model: Model, prebuilts: typeof SCRATCH_Prebuilts): Config[] => {
  const bias = buildBiases(model)

  return [
    {
      sampler: bias.accept(prebuilts.number)
    },
    {
      sampler: bias.prefer(oneOf(['boop', 'beep']), 100)
    }
  ]
}
