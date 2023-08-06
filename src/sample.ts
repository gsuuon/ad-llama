import type { NDArray, Instance } from 'tvmjs'

type Selector = (logits: NDArray, tokens: number[], completion: string) => number[]

type Bias = (selector: Selector, weight: number) => Sampler
type Gate = (selector: Selector) => Sampler

type Sampler = (logits: NDArray, tokens: number[], config: any) => number

type SamplerData = {
  vocab: string[] /// Vocab and vocab with leading whitespace (2x length of vocab)
  tvm: Instance
}

type SelectBuilder = (samplerData: SamplerData) => Selector
type Select = (selectBuilder: SelectBuilder) => Selector

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
  number: (samplerData: SamplerData) => {
    // TODO can V8 optimize this better with a for loop?
    const numberToks = samplerData.vocab.reduce<number[]>((acc, chars, idx) => {
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

      return (samplerData: SamplerData) => {

        return (_logits: NDArray, _tokens: number[], completion: string) => {
          let parsingTokens: number[] = []

          for (let idx = 0; idx < samplerData.vocab.length; idx++) {
            // TODO actual parser api
            parser.edit(completion + samplerData.vocab[idx])

            if (parser.parse()) {
              if (idx > samplerData.vocab.length) {
                parsingTokens.push(idx - samplerData.vocab.length)
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
  return {
    prefer: (selector: Selector, weight: number): Sampler => {
      return (logits: NDArray, tokens: number[], config: any) => {
        const relevantTokens = selector(logits, tokens, model.completion)

        model.applyRepetitionPenalty(logits, relevantTokens, 1/weight)
        return model.sampleNextToken(logits, config) as number
      }
    },
    avoid: (selector: Selector, weight: number): Sampler => {
      return (logits: NDArray, tokens: number[], config: any) => {
        const relevantTokens = selector(logits, tokens, model.completion)

        model.applyRepetitionPenalty(logits, relevantTokens, weight)
        return model.sampleNextToken(logits, config) as number
      }
    },
    accept: (selector: Selector): Sampler => {
      return (logits: NDArray, tokens: number[], config: any) => {
        const relevantTokens = selector(logits, tokens, model.completion)

        const modified = logits.toArray().map((logit, idx) => relevantTokens.includes(idx) ? logit : Number.NEGATIVE_INFINITY)

        logits.copyFrom(new Float32Array(modified))

        return model.sampleNextToken(logits, config) as number
      }
    },
    reject: (selector: Selector): Sampler => {
      return (logits: NDArray, tokens: number[], config: any) => {
        const relevantTokens = selector(logits, tokens, model.completion)

        const modified = logits.toArray().map((logit, idx) => relevantTokens.includes(idx) ? Number.NEGATIVE_INFINITY : logit)

        logits.copyFrom(new Float32Array(modified))

        return model.sampleNextToken(logits, config) as number
      }
    }
  }
}

export const fn = (model: any, select: Select, prebuilts: typeof SCRATCH_Prebuilts): Config[] => {
  const bias = buildBiases(model)

  return [
    {
      sampler: bias.prefer(select(prebuilts.number), 100)
    },
    {
      sampler: bias.accept(select(prebuilts.treesitter.json()))
    }
  ]
}
