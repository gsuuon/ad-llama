# Usage
Check public/index.html for an example

```javascript
import { initialize, ad, guessModelSpecFromPrebuiltId } from './index.js'

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
