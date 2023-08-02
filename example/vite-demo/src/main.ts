import './style.css'
import { ad, guessModelSpecFromPrebuiltId, loadModel } from 'ad-llama'

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div>
    <h1>ad-llama + vite HMR</h1>
    <pre>
      <code>
{
  "description": "\${(a('description'))}",
  "weapon": "\${a('weapon')}",
  "items": ["\${'three items in their possession'}]
}
      </code>
    </pre>
    <p id='text'></p>
  </div>
`

if (import.meta.hot) { import.meta.hot.accept() }

const gen = ad(await loadModel(guessModelSpecFromPrebuiltId('Llama-2-7b-chat-hf-q4f32_1')))

const { template, a } = gen('You are a dungeon master.', 'Create an interesting character based on the Dungeons and Dragons universe.')

const result = template`
{
  "description": "${(a('description', {maxTokens: 1000, stops: ['\n']}))}",
  "weapon": "${a('weapon')}",
  "items": ["${'three items in their possession'}]
}
`

const textEl = document.querySelector('#text')!

const text = await result.collect(partial => {
  const el = document.createElement('span')
  el.textContent = partial.content
  el.className = partial.type
  textEl.appendChild(el)
})

console.log(text)

console.log(JSON.parse(text))

