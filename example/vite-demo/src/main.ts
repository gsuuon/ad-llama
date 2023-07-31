import './style.css'
import { ad, guessModelSpecFromPrebuiltId, loadModel } from 'ad-llama'

if (import.meta.hot) { import.meta.hot.accept() }

const gen = ad(await loadModel(guessModelSpecFromPrebuiltId('Llama-2-7b-chat-hf-q4f32_1')))

const { template, a } = gen('<<sys>>You are a dungeon master. <</sys>>\n\n[INST] Create an interesting character based on the Dungeons and Dragons universe.')

const result = template`
{
  "description": "${(a('description'))}",
  "weapon": "${a('weapon')}",
  "items": [${a('three items in their possession')}]
}
`

const text = await result.collect()
console.log(text)

console.log(JSON.parse(text))

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div>
    <h1>ad-llama + vite HMR</h1>
    <p>${text}</p>
  </div>
`
