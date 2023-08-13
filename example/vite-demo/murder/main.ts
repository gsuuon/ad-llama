import './style.css'
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
const { consistsOf } = sample

// try to get a '\n' to stop at a paragraph
// TODO sampler which starts preferring '\n' after a certain amount of tokens
// for now we just retry if we got nothing
const breakParagraph = {
  sampler: bias.prefer(consistsOf(['\n']), 1.2),
  stops: ['\n'],
  validate: { check: (x: string) => x.length > 1, retries: 3 }
}

const backgroundInf = await renderTemplate(infer, async () => {
  const { template, a, __ } = createCtx('You are a text RPG game runner for a murder-mystery game.')

  return template`
  {
    "setting": "${__('Describe the world in which this game is set', {
      ...breakParagraph,
      id: 'setting',
      maxTokens: 750,
      validate: { check: x => x.length > 50, retries: 3 }, // sometimes we just get 'Ravenwood'
    })}",
    "numCharacters": 3,
    "characters": ["${a('list of the names of the non-player characters', {
      sampler: bias.reject(consistsOf(['\n', '{'])),
    })}]
  }`
})

const text = (body: string, className?: string) => {
  const el = document.createElement('p')
  if (className) {
    el.className = className
  }
  el.textContent = body
  return el
}

const background = JSON.parse(backgroundInf.completion)

const characterNames: string[] = background.characters
console.log({characterNames})

render(game, `
  <div id='scene'></div>
  <h2>Setting</h2>
  <p>${background.setting}</p>
  <h2>Characters</h2>
  <ul>
    ${characterNames.map(name => `<li id="${name}">${name}</li>`).join('')}
  </ul>
`)

let characters: any[] = []
for (const name of characterNames) {
  const { template, a } = createCtx(
    'You are a text RPG game runner for a murder-mystery game.',
    `Setting:
${background.setting}

Non-player characters:
 - ${characterNames.join('\n - ')}
`
  )
  // how do I put a literal string from previous?
  const characterInference = await renderTemplate(infer, async () => template`{
    "name": "${name}", 
    "role": "${a('role for this non-player character - what is their place in this story?')}",
    "description": "${a('description of this non-player character', {
      maxTokens: 600,
      ...breakParagraph
    })}",
    "secret": "${a('hidden secret about this character', breakParagraph)}",
    "motivation": "${a('motivation of this character which may hint at their secret', breakParagraph)}",
    "summary": "${a('one sentence summary of the character containing only publically known information about them', {stops:['.']})}"
  }`)

  const character = JSON.parse(characterInference.completion)
  characters.push(character)

  const charEl =document.getElementById(name)!
  charEl.textContent = charEl.textContent + ' - ' + character.role

  charEl.append(
    text(character.summary, 'summary'),
    text(character.description),
    text('Motivation: ' + character.motivation),
    text(character.secret, 'secret')
  )
}

console.log({characters})

const sceneInf = await renderTemplate(infer, async () => {
  const { template, a } = createCtx(
    'You are a text RPG game runner for a murder-mystery game.',
    `Setting:
${background.setting}

Non-player characters:
 - ${characters.map(char => char.summary).join('\n - ')}

Set the scene for where the player currently is.
`
  )

  return template`{
  "description": "${a('description of the current scene', breakParagraph)}",
  "characters": ["${a('list of the names of the non-player characters present which the player could talk to')}]
}`
})

const scene = JSON.parse(sceneInf.completion)

render(document.getElementById('scene')!, `
  <h2>Scene</h2>
  <p>${scene.description}</p>
  <h4>People present:</h4>
  <ul>
    ${scene.characters.map((name: string) => `<li id="${name}">${name}</li>`).join('')}
  </ul>
`)
