import { LoadedModel, sample } from 'ad-llama'

export const breakParagraph = (model: LoadedModel) => ({
  // try to get a '\n' to stop at a paragraph
  // TODO sampler which starts preferring '\n' after a certain amount of tokens
  // for now we just retry if we got nothing
  sampler: model.bias.prefer(sample.consistsOf(['\n']), 1.2),
  stops: ['\n'],
  validate: { check: (x: string) => x.length > 1, retries: 3 }
})
