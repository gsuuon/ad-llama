import './style.css'
import { renderTemplate } from './renderTemplate'
import { ad, guessModelSpecFromPrebuiltId, loadModel } from 'ad-llama'

if (import.meta.hot) { import.meta.hot.accept() }

const app = document.querySelector<HTMLDivElement>('#app')!

renderTemplate(app, async () => {
  const gen = ad(
    await loadModel(
      guessModelSpecFromPrebuiltId('Llama-2-7b-chat-hf-q4f32_1'),
      undefined,
      report => app.innerHTML = `<pre id='progress'><code>${JSON.stringify(report, null, 2)}</code></pre>`
    )
  )

  const { template, a } = gen(
    'You are a dungeon master.',
    'Create an interesting character based on the Dungeons and Dragons universe.'
  )
  return template`{
    "description": "${(a('description', {maxTokens: 1000, stops: ['\n']}))}",
    "name": "${(a('name'))}",
    "weapon": "${a('weapon')}",
    "items": [
      {
        "name": "${a('name')}",
        "description": "${a('short description')}",
        "type": "${a('type')}"
      },
      {
        "name": "${a('name')}",
        "description": "${a('short description')}",
        "type": "${a('type')}"
      },
      {
        "name": "${a('name')}",
        "description": "${a('short description')}",
        "type": "${a('type')}"
      }
    ]
  }`
})
