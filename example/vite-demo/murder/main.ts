import '../style.css'
import { renderTemplate } from '../renderTemplate'
import { TargetDevice, ad, loadModel, sample } from 'ad-llama'

if (import.meta.hot) { import.meta.hot.accept() }

const infer = document.querySelector<HTMLDivElement>('#infer')!
const game = document.querySelector<HTMLDivElement>('#game')!

const render = (el: HTMLElement, html: string) => el.innerHTML = html

const model = await loadModel(
  'Llama-2-7b-chat-hf-q4f32_1',
  report => infer.innerHTML = `<pre id='progress'><code>${JSON.stringify(report, null, 2)}</code></pre>`,
  new URLSearchParams(window.location.search).get('cpu') === null
    ? TargetDevice.GPU
    : TargetDevice.CPU
)

const createCtx = ad(model)

const { bias } = model
const { consistsOf, chars } = sample

const setting = await renderTemplate(infer, async () => {
  const { template, a, __ } = createCtx('You are a text RPG game runner for a murder-mystery game.')

  return template`
  {
    "setting": "${a('setting for this session', {
      id: 'setting',
      maxTokens: 700,
      // try to get a '\n' to stop at a paragraph
      sampler: bias.prefer(consistsOf(['\n']), 1.1),
      stops: ['\n'],
      validate: { check: x => x.length > 50, retries: 3 } // sometimes we just get 'Ravenwood'
    })}",
    "numCharacters": ${__('Decide the number of non-player characters for this session', {
    sampler: bias.accept(chars.number)
  })},
  "characters": ["${a('list of the names of the non-player characters', {
    id: 'characters',
    sampler: bias.reject(consistsOf(['\n', '{'])),
  })}]
}`
})

const text = (body: string) => {
  const el = document.createElement('p')
  el.textContent = body
  return el
}

const characterNames: string[] = JSON.parse(`["${setting.refs['characters']!}]`)
console.log({characterNames})

render(game, `
  <h2>Setting</h2>
  <p>${setting.refs['setting']}</p>
  <h2>Characters</h2>
  <ul>
    ${characterNames.map(name => `<li id="${name}">${name}</li>`).join('')}
  </ul>
`)

let characters = []
for (const name of characterNames) {
  const { template, a } = createCtx(
    'You are a text RPG game runner for a murder-mystery game.',
    `Setting: ${setting.refs['setting']}\n\n`
  )
  // how do I put a literal string from previous?
  const characterInference = await renderTemplate(infer, async () => template`{
    "name": "${name}", 
    "description": "${a('description of this character', { maxTokens: 600, stops: ['\n'] })}",
    "motivation": "${a('motivation of this character')}",
    "summary": "${a('one sentence summary')}"
  }`)

  const character = JSON.parse(characterInference.completion)
  characters.push(character)

  const charEl =document.getElementById(name)!

  charEl.append(
    text(character.description),
    text(character.motivation),
    text(character.summary)
  )
}

console.log({characters})
