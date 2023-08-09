import './style.css'
import { renderTemplate } from './renderTemplate'
import { ad, guessModelSpecFromPrebuiltId, loadModel, TargetDevice, chars, oneOf } from 'ad-llama'

if (import.meta.hot) { import.meta.hot.accept() }

const app = document.querySelector<HTMLDivElement>('#app')!

const alsoToLowerCase = (x: string) => [x.toLowerCase(), x]

renderTemplate(app, async () => {
  const model =
    await loadModel(
      guessModelSpecFromPrebuiltId('Llama-2-7b-chat-hf-q4f32_1'),
      report => app.innerHTML = `<pre id='progress'><code>${JSON.stringify(report, null, 2)}</code></pre>`,
      new URLSearchParams(window.location.search).get('cpu') === null
        ? TargetDevice.GPU // FIXME TargetDevice.CPU seems to be broken right now?
        : TargetDevice.CPU 
    )

  const gen = ad(model)

  const { template, a } = gen(
    'You are a dungeon master.',
    'Create an interesting character based on the Dungeons and Dragons universe.'
  )

  return template`{
  "class": "${a('class', {
    sampler: model.biases.reject(oneOf(['Ranger', 'Rogue'].flatMap(alsoToLowerCase)))
  })}",
  "subclass": "${a('subclass')}",
  "name": "${(a('name'))}",
  "weapon": "${a('special weapon', {
    sampler: model.biases.prefer(oneOf(['nun-chucks', 'beam cannon']), 100)
  })}",
  "description": "${(a('clever description', {maxTokens: 1000, stops: ['\n']}))}",
  "age": ${a('age', {
    sampler: model.biases.accept(chars.number([','])),
    maxTokens: 4,
  })},
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
}`})
