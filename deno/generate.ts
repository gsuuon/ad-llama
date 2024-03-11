import { loadModel, ad, sample } from "../src/index.ts";

console.log("Loading");

const model = await loadModel("Llama-2-7b-chat-hf-q4f16_1", console.log);

console.info("Loaded");

const inferStarWars = async () => {
  const { a, context } = ad(model);

  const assistant = context("You are a helpful assistant");

  console.info("Inferencing");

  const template = assistant`{ "name" : "${a("Star Wars character name", {
    sampler: model.bias.accept(sample.oneOf(['Beepity"', 'Boopity"'])),
  })}"}`;

  const result = await template.collect((partial) =>
    console.info(JSON.stringify(partial.content)),
  );

  console.log({ result });
};

await inferStarWars();
