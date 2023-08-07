import { NDArray, Instance } from 'tvmjs'
import type { Tokenizer } from '@mlc-ai/web-tokenizers'

type Selector = (logits: NDArray, tokens: number[], completion: string) => number[]

type Bias = (selector: SelectBuilder, weight: number) => Sampler
type Mask = (selector: SelectBuilder) => Sampler

type Sampler = (logits: NDArray, tokens: number[], completion: string, config: any) => number

type Model = {
  vocab: string[] /// Vocab and vocab with leading whitespace (2x length of vocab)
  tvm: Instance
  tokenizer: Tokenizer
}

type SelectBuilder = (model: Model) => Selector

type Biases = {
  prefer: Bias
  avoid: Bias
  accept: Mask
  reject: Mask
}

type Config = {
  sampler: Sampler
}

const SCRATCH_Prebuilts = {
  number: (model: Model) => {
    // TODO can V8 optimize this better with a for loop?
    const numberToks = model.vocab.reduce<number[]>((acc, chars, idx) => {
      if (!isNaN(Number(chars))) {
        acc.push(idx)
      }
      return acc
    }, [])

    return () => numberToks
  },
  treesitter: {
    json: () => {
      const parser: any = {}

      return (model: Model) => {

        return (_logits: NDArray, _tokens: number[], completion: string) => {
          let parsingTokens: number[] = []

          for (let idx = 0; idx < model.vocab.length; idx++) {
            // TODO actual parser api
            parser.edit(completion + model.vocab[idx])

            if (parser.parse()) {
              if (idx > model.vocab.length) {
                parsingTokens.push(idx - model.vocab.length)
              } else {
                parsingTokens.push(idx)
              }
            }

            parser.edit(completion)
          }

          return parsingTokens
        }
      }
    }
  }
}

const buildBiases = (model: Model): Biases => {
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

      const modified = logits.toArray().map(adjust(relevantTokens))

      logits.copyFrom(new Float32Array(modified))

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

const arrayStartsWith = <T>(starts: T[]) => (xs: T[]) => {
  for (let i = 0; i < starts.length; i++) {
    if (starts[i] !== xs[i]) {
      return false
    }
  }

  return true
}

const oneOf = (items: string[]): SelectBuilder => {
  return (model: Model) => {
    const encoded: number[][] = items.map(x => Array.from(model.tokenizer.encode(x)))

    return (_logits, tokens, _completions) => {
      return encoded.filter(arrayStartsWith(tokens)).map(x => x[tokens.length] )
    }
  }
}

export const fn = (model: any, prebuilts: typeof SCRATCH_Prebuilts): Config[] => {
  const bias = buildBiases(model)

  return [
    {
      sampler: bias.prefer(prebuilts.number, 100)
    },
    {
      sampler: bias.accept(prebuilts.treesitter.json())
    },
    {
      sampler: bias.prefer(oneOf(['boop', 'beep']), 100)
    }
  ]
}
