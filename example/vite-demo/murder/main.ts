import './style.css'
import { renderTemplate } from '../renderTemplate'
import { TargetDevice, ad, loadModel, sample } from 'ad-llama'
import { establishBackgroundAndCharacters } from './inferBackground'
import checkpoint from './story_checkpoint.json'

if (import.meta.hot) { import.meta.hot.accept() }

const infer = document.querySelector<HTMLDivElement>('#infer')!

const model = await loadModel(
  'Llama-2-7b-chat-hf-q4f32_1',
  report => infer.innerHTML = `<pre id='progress'><code>${JSON.stringify(report, null, 2)}</code></pre>`,
  new URLSearchParams(window.location.search).get('cpu') === null
    ? TargetDevice.GPU
    : TargetDevice.CPU
)

// const { characters, background, scene } = await establishBackgroundAndCharacters(model)
const { characters, background, scene } = checkpoint
console.log({ characters, background, scene })

const createCtx = ad(model)

const { bias } = model
const { consistsOf, oneOf } = sample

const o = (parent: HTMLElement, children: HTMLElement[]) => {
  parent.append(...children)
  return parent
}

const el = (tag: string, fn?: (el: any) => void) => {
  const e = document.createElement(tag)
  fn?.(e)
  return e
}

// const actionsEl = document.getElementById('actions')!
const actionsEl = document.getElementById('infer')!
const getUserAction = (): Promise<string> => new Promise(resolve => {
  const form = el('form') as HTMLFormElement
  const submit = el('button', x => { x.type = 'submit'; x.innerText = '>'})

  form.onsubmit = (ev: any) => {
    ev.preventDefault()
    const value = ev.target[0].value

    if (value) {
      submit.remove()
      resolve(value)
    }
  }

  actionsEl.prepend(o(
    form,
    [
      el('label', x => x.innerText = 'action'),
      el('input', x => x.type = 'text'),
      submit
    ]
  ))
})

const { template, a } = createCtx(
  'You are a text RPG game runner for a murder-mystery game.',
  'Setting:\n'
  + background.setting
  + '\nNon-player characters:n'
  + characters.map(char => `${char.name} -- ${char.summary}`).join('\n - ')
)

while (true) {
  // const playerAction_ = await getUserAction()
  const playerAction_ = 'I walk up to Emily and ask, "What do you make of this case?"'

  const playerAction = playerAction_.replace(/"/g, '\\"')

  const action: {
    playerAction: string,
    relevantCharacters: string[],
    actionType: string
  } = JSON.parse(await renderTemplate(infer, async () => template`{
    "playerAction": "${playerAction}",
    "relevantCharacters": [${a('list of zero or more names of the most relevant non-player characters given the player action', { sampler: bias.reject(consistsOf(['\n', '{']))})}],
    "actionType": "${a('type for the action', {sampler: bias.accept(oneOf(['talk', 'move', 'attack', 'act']))})}"
  }`, false))

  const relevantChars = action.relevantCharacters.map((name: string) => characters.find(char => char.name.toLowerCase().includes(name.toLowerCase()))).filter(x => x !== undefined)

  console.log({relevantChars})

  const startToks = model.totalTokenCount

  const step = await renderTemplate(infer, async () => template`
Player: ${playerAction_}

Narrator:
${a('description of the results of what the player does. If an npc responds, respond as that character.', {
stops: ['Player:'],
maxTokens: 2000
})}`, false)

  console.info('step', {stepTokens: model.totalTokenCount - startToks})

  break
}
