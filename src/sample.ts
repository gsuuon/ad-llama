import type { NDArray, Instance } from 'tvmjs'

type Selector = (logits: NDArray, tokens: number[]) => number[]

type Bias = (select: Select, weight: number) => Sampler
type Gate = (select: Select) => Sampler

type Sampler = (logits: NDArray, tokens: number[]) => number

type Compiler = (vocab: string[], vocabWhitespace: string[]) => Selector

type DynamicSampler = (vocab: string[], vocabWhitespace: string[], tvm: Instance) => Selector

type Select = {
  statically: (choices: string[]) => Selector
  precompiled: (compiler: Compiler) => Selector
  dynamically: (dynamicSampler: DynamicSampler) => Selector
}

type Biases = {
  prefer: Bias
  avoid: Bias
  accept: Gate
  reject: Gate
}
