# ad-llama 🦙

Use tagged template literals for structured LLaMa 2 inference locally in browser via [mlc-llm](https://github.com/mlc-ai/mlc-llm). Runs on Chromium browsers with WebGPU support. Check [example/vite-demo](./example/vite-demo) for an example or https://ad-llama.vercel.app/ for a live demo.

https://github.com/gsuuon/ad-llama/assets/6422188/35a1316a-c491-4163-a3c7-cb8bafe74bff

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
