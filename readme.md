# ad-llama 🦙

Use tagged template literals for structured LLaMa 2 inference locally in browser via [mlc-llm](https://github.com/mlc-ai/mlc-llm). Runs on Chromium browsers with WebGPU support. Check [example/vite-demo](./example/vite-demo) for an example or https://ad-llama.vercel.app/ for a live demo.

https://github.com/gsuuon/ad-llama/assets/6422188/54fed226-c29b-44d6-a797-cc39a4e5a5d1

Say hi in [discord](https://discord.gg/Jag2h3fS4C)!

# Usage
`npm install -S ad-llama`

```javascript
import { loadModel, ad, guessModelSpecFromPrebuiltId } from 'ad-llama'

const loadedModel = await loadModel(guessModelSpecFromPrebuiltId('Llama-2-7b-chat-hf-q4f32_1'))
const generator = ad(loadedModel)

const { template, a } = generator(
  'You are a dungeon master.',
  'Create a character based on the Dungeons and Dragons universe.'
)

const result = template`
{
  "description": "${a('short description')}",
  "name": "${a('character name')}",
  "weapon": "${a('weapon')}",
  "class": "${a('class')}"
}
`

console.log(await result.collect())
```

For an example of more complicated usage including validation, retry logic and transforms check the hackernews [who's hiring example.](./example/vite-demo/hn/main.ts)

## API
### Generation
Each expression in the template literal can be configured independently - you can set a different temperature, token count, max length and more. Check the `AdExprConfig` type in [./src/types.ts](./src/types.ts) for all options. `a` adds the preword to the expression prompt (by default "Generate a"), you can use `__` to provide a naked prompt or configure the preword as needed. If you don't need to set expression options at all, just put a string in the expression.

```typescript
template`{
  "description": "${a('clever description', {
    maxTokens: 1000,
    stops: ['\n']
  })}",
  "class": "${'a primary class for the character'}"
}`
```

#### Biased Sampling
You can modify the sampler for each expression -- this allows you to adjust the logits before sampling. You can, for example, only accept number characters for one expression, while in another avoid specific strings. The main example [here](./example/vite-demo/src/main.ts) shows some of these options. You can build your own, but there are some helper functions exposed as `sample` to build samplers. The loaded `model` object has a `bias` field which configures the sampling - `avoid`, `prefer` allow you to adjust relative likelihood of certain tokens ids, while `reject`, `accept` change the logits to negative infinity for some ids (or all other ids).

```typescript
import { sample } from 'ad-llama'

const model = await loadModel(...)

const { bias } = model
const { oneOf, consistsOf } = sample

template`{
  "weapon": "${a('special weapon', {
    sampler: bias.prefer(oneOf(['Nun-chucks', 'Beam Cannon']), 10),
  })}",
  "age": ${a('age', {
    sampler: bias.accept(consistsOf(['0','1','2','3','4','5','6', '7', '8', '9'])),
    maxTokens: 3
  })},
}`
```

##### Sampler builder helpers

`oneOf`, `consistsOf` try to generate relevant tokens for the provided strings in the context of the current generation expression -- as tokenizers are stateful, a simple encoding of just the provided strings won't necessarily produce tokens that would fit into the existing sequence. For example, with Llama 2's tokenizer `foo` and `"foo"` encode to completely different tokens:

```javascript
encode('"foo"') === [376, 5431, 29908]
encode('foo') === [7953]
decode([5431]) === 'foo'
decode([7953]) === 'foo'
```

`consistsOf` is for classes of characters - each sample in that expression generation will have the same token logits modified based on the given relevant tokens. So in `consistsOf(['a','b'])`, every sample will have tokens for 'a' and 'b' modified.

`oneOf` is for strings - each sample has logits modified depending on the tokens which are still relevant given the already sampled tokens. For example, if we have `oneOf(['ranger', 'wizard'])` and we've already sampled 'w', the only next relevant tokens would be from 'izard'. If you want `oneOf` to stop at one of the choices, include the stop character (by default the next character after the expression), eg: `oneOf(['ranger"', 'wizard"'])`.

Even though `reject(oneOf(['ranger', 'wizard']))` will never make it past the first token for either of the strings, giving the entire string still lets you target the correct tokens for completing those specific strings.

#### Validation
You can provide an expression validation function with a retry count. If validation fails, that expression will be attempted again up to retry times, after which whatever was generated is taken. You can also transform the result of the expression generation (this happens whether validation passes or not).

```typescript
validate?: {
  check?: (partial: string) => boolean
  transform?: (partial: string) => string
  retries: number
}
```

## Vite HMR
Waiting for models to reload can be tedious, even when they're cached. ad-llama should work with vite HMR so the loaded models stay in memory. Put this in your source file to create an HMR boundary:
```diff
import { loadModel, ad, guessModelSpecFromPrebuiltId } from 'ad-llama'

+ if (import.meta.hot) { import.meta.hot.accept() }

const loadedModel = await loadModel(guessModelSpecFromPrebuiltId('Llama-2-7b-chat-hf-q4f32_1'))
```


# Build
- pre-reqs
  - emcc: https://emscripten.org/docs/getting_started/downloads.html
- build the tvmjs dependency first
  ```bash
  git clone https://github.com/gsuuon/ad-llama.git --recursive
  cd 3rdparty/relax/web

  # source ~/emsdk/emsdk_env.sh
  make
  npm install
  npm run build
  ```
- then either `npm run build` or `npm run dev` (which watches `src/` and serves `public/`)

# Motivation
I was inspired by [guidance](https://github.com/microsoft/guidance) but felt that tagged template literals were a better way to express structured inference. I also think [grammar](https://github.com/ggerganov/llama.cpp/pull/1773) based sampling is neat, and wanted to add a way to plug something like that into MLC infrastructure.

# Todos
- [ ] runs on Deno
- [ ] can target cpu
- [ ] repeatable subtemplates
- [ ] template expressions can reference previously generated expressions
