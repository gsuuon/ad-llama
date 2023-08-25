# ad-llama 🦙


Use tagged template literals for structured Llama 2 inference locally in browser via [mlc-llm](https://github.com/mlc-ai/mlc-llm). Runs on Chromium browsers with WebGPU support. 


https://github.com/gsuuon/ad-llama/assets/6422188/d38ef8c3-8ba1-4757-a992-ea8720776780


Check out the [playground](https://ad-llama.vercel.app/playground/) or a static [demo](https://ad-llama.vercel.app/). There's also a [documentation](https://gsuuon.github.io/ad-llama/) page --
say hi in [discord](https://discord.gg/Jag2h3fS4C)!

# Usage
`npm install -S ad-llama`

```javascript
import { loadModel, ad, report } from 'ad-llama'

const loadedModel = await loadModel('Llama-2-7b-chat-hf-q4f32_1', report(console.info))
const { context, a } = ad(loadedModel)

const dm = context(
  'You are a dungeon master.',
  'Create an NPC based on the Dungeons and Dragons universe.'
)

const npc = dm`{
  "description": "${a('short description')}",
  "name": "${a('character name')}",
  "weapon": "${a('weapon')}",
  "class": "${a('primary class')}"
}`

const generatedNpc = await npc.collect(partial => {
  switch (partial.type) {
    case 'gen':
    case 'lit':
      console.info(partial.content)
      break
  }
})

console.log(generatedNpc)
```

For an example of more complicated usage including validation, retry logic and transforms check the hackernews [who's hiring example.](https://github.com/gsuuon/ad-llama/tree/main/example/vite-demo/hn/main.ts)

## Generation
Each expression in the template literal is a new prompt and options. The prompt given for each expression is added to the system and preprompt established in context, and prior completion text (literal parts and as well as inferences) are added to the end of the LLM prompt as a partially completed assistant response (i.e. after [/INST]).

Each template expression can be configured independently - you can set a different temperature, token count, max length and more. Check the `TemplateExpressionOptions` type in [./src/types.ts](https://github.com/gsuuon/ad-llama/tree/main/src/types.ts) for all options. `a` adds the preword to the expression prompt (by default "Generate a"), you can use `__` to provide a naked prompt, or configure the preword as needed. A plain string gets inserted as literal text, just like normal template literals.

```typescript
template`{
  "name": "${characterName}",
  "description": "${a('clever description', {
    maxTokens: 1000,
    stops: ['\n']
  })}",
  "class": "${a('a primary class for the character')}"
}`
```

### Biased Sampling
You can modify the sampler for each expression -- this allows you to adjust the logits before sampling. You can, for example, only accept number characters for one expression, while in another avoid specific strings. The main example [here](https://github.com/gsuuon/ad-llama/tree/main/example/vite-demo/src/main.ts) shows some of these options. You can build your own, but there are some helper functions exposed as `sample` to build samplers. The loaded `model` object has a `bias` field which configures the sampling - `avoid`, `prefer` allow you to adjust relative likelihood of certain tokens ids, while `reject`, `accept` change the logits to negative infinity for some ids (or all other ids).

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

As tokenizers are stateful, a simple encoding of just the provided strings won't necessarily produce tokens that would fit into the existing sequence. For example, with Llama 2's tokenizer `foo` and `"foo"` encode to completely different tokens:

```javascript
encode('"foo"') === [376, 5431, 29908]
encode('foo') === [7953]
decode([5431]) === 'foo'
decode([7953]) === 'foo'
```

`oneOf`, `consistsOf` try to figure out relevant tokens for the provided strings within the context of the currently inferring expression.

`consistsOf` is for classes of characters - each sample will have the same token logits modified based on the given relevant tokens. So in `consistsOf(['a','b'])`, every sample will have tokens for 'a' and 'b' modified.

`oneOf` is for strings - each sample has logits modified depending on the tokens which are still relevant given the already sampled tokens. For example, if we have `oneOf(['ranger', 'wizard'])` and we've already sampled 'w', the only next relevant tokens would be from 'izard'. If you want `oneOf` to stop at one of the choices, include the stop character (by default the next character after the expression), eg: `oneOf(['ranger"', 'wizard"'])`.

Even though `reject(oneOf(['ranger', 'wizard']))` will never make it past the first token for either of the strings, giving the entire string still lets you target the correct tokens for completing those specific strings.

### Validation
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
