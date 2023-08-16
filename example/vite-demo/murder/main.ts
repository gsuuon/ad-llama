import './style.css'
import { renderTemplate, renderTemplateRefs } from '../renderTemplate'
import { TargetDevice, ad, loadModel, sample } from 'ad-llama'
import { establishBackgroundAndCharacters } from './inferBackground'
// import checkpoint from './story_checkpoint.json'

if (import.meta.hot) { import.meta.hot.accept() }

const infer = document.getElementById('infer')!

const model = await loadModel(
  'Llama-2-7b-chat-hf-q4f32_1',
  report => infer.innerHTML = `<pre id='progress'><code>${JSON.stringify(report, null, 2)}</code></pre>`,
  new URLSearchParams(window.location.search).get('cpu') === null
    ? TargetDevice.GPU
    : TargetDevice.CPU
)

const { characters, background, scene } = await establishBackgroundAndCharacters(model)
// const { characters, background, scene } = checkpoint
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

const showAndWaitUserInput = (): Promise<string> => new Promise(resolve => {
  const form = el('form') as HTMLFormElement
  const submit = el('button', x => { x.type = 'submit'; x.innerText = '>'})

  form.onsubmit = (ev: any) => {
    ev.preventDefault()
    const value = ev.target[0].value

    if (value) {
      form.remove()
      resolve(value)
    }
  }

  infer.prepend(o(
    form,
    [
      el('label', x => x.innerText = 'action'),
      el('input', x => x.type = 'text'),
      submit
    ]
  ))
})

const { template: storyRunner, __, a } = createCtx(
  'You are a story generator.',
  '\nSetting:\n'
  + background.setting
  + '\n\nNon-player characters:\n'
  + characters.map(char => ` - ${char.name} -- ${char.summary}`).join('\n')
  + '\n\n'
)

const findNpc = (name: string) => characters.find(char => char.name.toLowerCase().includes(name.toLowerCase()))

type PlayerActionType = "interactNpc" | "generalAction"

const parseInputType = async (input: string) => {
  const { template, a } = createCtx(
    'You are a text role-playing game runner.', `
Player input: ${input}

Choose which type of action the player input is:
 - interactNpc -- the player intends to interact with a character
 - generalAction -- the player is taking a general action with no specific character in mind
`, { preword: 'What is' })

  return JSON.parse(await renderTemplate(infer, async () =>
    template`"${a('type for the action given the player input?', {
      sampler: bias.accept(oneOf(['interactNpc"', 'generalAction"'])),
      temperature: 0.2
    })}"`, false)) as PlayerActionType
}

const characterActionDefinition = `
\`\`\`ts
type CharacterAction = {
  said?: string // anything the character said
  did?: string // anything the character did
}
\`\`\`
`

const parseNarration = async (narration: string, knownCharacters: string[], relevantCharacter?: string) => {
  const { template, __ } = createCtx(
    'You are a story generator.',
    '\nNarration: """\n'
    + narration
    + '\n"""\n',
    { preword: 'What is' }
  )

  const character: string =
    relevantCharacter ?? JSON.parse(await renderTemplate(infer, async () => template`"${
      __(
        'Among these characters:\n'
        + characters.map(char => ` - ${char.name} -- ${char.summary}`).join('\n')
        + '\n\nWhich character is this narration about?',
        {
          sampler: bias.accept(oneOf(knownCharacters.map(name => `"${name}"`))),
          temperature: 0.2
        }
      )}"`)
    )

  console.log({relevantCharacters: character})

  const actions = JSON.parse(await renderTemplate(infer, async() =>
    template`[
  {
    "${__(`What is a list of actions of type CharacterAction that the character ${character} took?\n${characterActionDefinition}`, {
        sampler: bias.prefer(consistsOf(['}']), 1.3), // try to get an ending } without ,
        temperature: 0.2
      })}]`
  ))

  return actions
}

const stepInteractNpc = async (playerInput: string) => {
  const action: {
    playerInput: string,
    primaryNpc: string
    otherNpcs: string[],
    actionType: string
  } = JSON.parse(await renderTemplate(infer, async () => storyRunner`{
    "playerInput": ${JSON.stringify(playerInput)},
    "primaryNpc": "${__('Who is the primary character the action is directed to?', {
      sampler: bias.accept(oneOf(characters.map(c => c.name))),
      temperature: 0.2
  })}"
  }`, false))

  const relevantNpc = findNpc(action.primaryNpc)

  const npcDescription = relevantNpc ? `Relevant character: ${relevantNpc.name} (${relevantNpc.role})
${relevantNpc.description}
Secret information:
${relevantNpc.secret}` : ''

  const step = await renderTemplateRefs(infer, async () => storyRunner`
[Player] ${action.playerInput}

[${relevantNpc!.name}] ${__(
npcDescription + '\nGenerate a narration of what the character does in response to the player. Stick to a third person perspective.',
{ id: 'narration', stops: ['['] }
)}`, false)

  return { narration: step.refs.narration, character: relevantNpc?.name }
}

const stepActionInWorld = async (playerInput: string) => {
  const step = await renderTemplateRefs(infer, async () => storyRunner`[Player] ${playerInput }
[Narrator]
${a('narration of what the player does and the consequences.',
{ id: 'narration', stops: ['['] }
)}`, false)

  return { narration: step.refs.narration }
}

const step:
  Record<
    PlayerActionType,
    ( input: string) => Promise<{ narration: string, character?: string }>
  > = {
  generalAction: stepActionInWorld,
  interactNpc: stepInteractNpc
}

while (true) {
  const playerInput = await showAndWaitUserInput()

  const actionType = await parseInputType(playerInput)
  console.log({actionType})

  const { narration, character } = await step[actionType](playerInput)
  console.log({narration})

  const parsedNarration = await parseNarration(
    narration,
    characters.map(c => c.name),
    character
  )

  console.log({parsedNarration})
}
