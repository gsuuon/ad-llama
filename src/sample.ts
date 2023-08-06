import type { NDArray, Instance } from 'tvmjs'
import type { Tokenizer } from '@mlc-ai/web-tokenizers'

type Selector = (logits: NDArray, tokens: number[], completion: string) => number[]

type Bias = (selector: SelectBuilder, weight: number) => Sampler
type Gate = (selector: SelectBuilder) => Sampler

type Sampler = (logits: NDArray, tokens: number[], config: any) => number

type SelectorData = {
  vocab: string[] /// Vocab and vocab with leading whitespace (2x length of vocab)
  tvm: Instance
  tokenizer: Tokenizer
}

type SelectBuilder = (selectorData: SelectorData) => Selector

type Biases = {
  prefer: Bias
  avoid: Bias
  accept: Gate
  reject: Gate
}

type Config = {
  sampler: Sampler
}

const SCRATCH_Prebuilts = {
  number: (selectorData: SelectorData) => {
    // TODO can V8 optimize this better with a for loop?
    const numberToks = selectorData.vocab.reduce<number[]>((acc, chars, idx) => {
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

      return (selectorData: SelectorData) => {

        return (_logits: NDArray, _tokens: number[], completion: string) => {
          let parsingTokens: number[] = []

          for (let idx = 0; idx < selectorData.vocab.length; idx++) {
            // TODO actual parser api
            parser.edit(completion + selectorData.vocab[idx])

            if (parser.parse()) {
              if (idx > selectorData.vocab.length) {
                parsingTokens.push(idx - selectorData.vocab.length)
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

const buildBiases = (model: any): Biases => {
  const selectorData: SelectorData = {
    vocab: model.vocab,
    tvm: model.tvm,
    tokenizer: model.tokenizer
  }

  const penalize = (selectBuilder: SelectBuilder, weight: number) => {
    const selector = selectBuilder(selectorData)

    return (logits: NDArray, tokens: number[], config: any) => {
      const relevantTokens = selector(logits, tokens, model.completion)

      model.applyRepetitionPenalty(logits, relevantTokens, weight)
      return model.sampleNextToken(logits, config) as number
    }
  }

  const gate = (selectBuilder: SelectBuilder, adjust: (relevantTokens: number[]) => (logit: number, idx: number) => number) => {
    const selector = selectBuilder(selectorData)

    return (logits: NDArray, tokens: number[], config: any) => {
      const relevantTokens = selector(logits, tokens, model.completion)

      const modified = logits.toArray().map(adjust(relevantTokens))

      logits.copyFrom(new Float32Array(modified))

      return model.sampleNextToken(logits, config) as number
    }
  }

  return {
    prefer: (selectBuilder: SelectBuilder, weight: number): Sampler => penalize(selectBuilder, 1/weight),
    avoid: (selectBuilder: SelectBuilder, weight: number): Sampler => penalize(selectBuilder, weight),
    accept: (selectBuilder: SelectBuilder): Sampler => gate(
      selectBuilder,
      (relevantTokens) => (logit, idx) => relevantTokens.includes(idx) ? logit : Number.NEGATIVE_INFINITY
    ),
    reject: (selectBuilder: SelectBuilder): Sampler => gate(
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
  return (selectorData: SelectorData) => {
    const encoded: number[][] = items.map(x => Array.from(selectorData.tokenizer.encode(x)))

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
