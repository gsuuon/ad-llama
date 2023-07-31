# Usage
Check public/index.html for an example

```javascript
import { initialize, ad, guessModelSpecFromPrebuiltId } from 'ad-llama'

const api = await initialize()
const loadedModel = await api.loadModel(guessModelSpecFromPrebuiltId('Llama-2-7b-chat-hf-q4f32_1'))
const generator = ad(loadedModel)

const { template, a } = generator(
  '<<sys>>You are a dungeon master. <</sys>>\n\n[INST] Create a character based on the Dungeons and Dragons universe.'
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


# Build
Requirements:
- emcc
    - https://emscripten.org/docs/getting_started/downloads.html

```bash
git clone .. --recursive
cd 3rdparty/relax/web

// source ~/emsdk/emsdk_env.sh
make
```

# Motivation
I was inspired by [guidance](https://github.com/microsoft/guidance) but thought that tagged template literals were a better way to express structured inference. I also thought [grammar](https://github.com/ggerganov/llama.cpp/pull/1773) based sampling was neat, and wanted to add a way to plug something like that into MLC infrastructure.
