import './style.css'
import { renderTemplate } from './renderTemplate'
import { ad, loadModel, TargetDevice, sample } from 'ad-llama'

if (import.meta.hot) { import.meta.hot.accept() }

const app = document.querySelector<HTMLDivElement>('#app')!

const alsoToLowerCase = (x: string) => [x.toLowerCase(), x]

renderTemplate(app, async () => {
  const model =
    await loadModel(
      'Llama-2-7b-chat-hf-q4f16_1',
      report => app.innerHTML = `<pre id='progress'><code>${JSON.stringify(report, null, 2)}</code></pre>`,
      new URLSearchParams(window.location.search).get('cpu') === null
        ? TargetDevice.GPU // TODO TargetDevice.CPU wont work until we have tvmjs wasm64 -- try to rebuild with emcc -sMEMORY64
        : TargetDevice.CPU
    )

  const { context, a } = ad(model)

  const dm = context(
    'You are a dungeon master.',
    'Create an interesting character based on the Dungeons and Dragons universe.'
  )

  const { bias } = model
  const { oneOf, consistsOf, chars } = sample

  return dm`{
  "class": "${a('primary class for the character', {
    sampler: bias.reject(oneOf(['Ranger', 'Rogue'].flatMap(alsoToLowerCase)))
  })}",
  "subclass": "${a('subclass')}",
  "name": "${(a('name', { sampler: bias.avoid(oneOf(['Eira', 'Zorvath', 'Kaelith']), 1.5) }))}",
  "weapon": "${a('special weapon', { sampler: bias.prefer(oneOf(['Nun-chucks', 'Beam Cannon']), 10) })}",
  "description": "${(a('clever description', {
    maxTokens: 1000,
    stops: ['\n'],
    sampler: bias.avoid(consistsOf(['\n']), 1.2)
  }))}",
  "age": ${a('age', {
    sampler: bias.accept(chars.number),
    maxTokens: 3
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
