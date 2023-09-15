import { For, createSignal } from 'solid-js'

import { ModelScene } from '../model'
import ShowInfer from '../component/ShowInfer'
import { breakParagraph } from '../llm/util'

import { View } from './type'

import './scene.css'


const view: View<ModelScene> = {
  "scene generate": ({ update, model, llm, context, a }) => {
    const { background, characters } = model()

    const gameRunner = context(
      'You are a text RPG game runner for a murder-mystery game.',
      'Setting:\n'
      + background.setting
      + '\nNon-player characters:\n'
      + characters.map(char => char.summary).join('\n - ')
      + '\nSet the scene for where the player currently is:')

    const [template] = createSignal(gameRunner`{
      "description": "${a('description of the current scene', breakParagraph(llm))}",
      "characters": ["${a('list of the names of the non-player characters present which the player could talk to')}]
    }`)

    return <ShowInfer
      template={template}
      onComplete={
        result => {
          const sceneGen: {
            description: string
            characters: string[]
          } = JSON.parse(result.completion)

          const scene = {
            description: sceneGen.description,
            characterNames: sceneGen.characters,
            conversations: []
          }

          console.log({ scene })

          update({
            state: 'scene player input',
            background,
            characters,
            scene,
            scenes: [scene]
          })
        }
      }
    />
  },
  "scene player input": ({model}) => {
    const [input, setInput] = createSignal('')
    const { background, scene } = model()

    return (
      <>
        <div>{background.setting}</div>
        <div>{scene.description}</div>
        <ul>
          <For each={scene.characterNames}>
            {name => <li>{name}</li>}
          </For>
        </ul>
        <div>
          <form onSubmit={e => {e.preventDefault(); console.log({input: input()})}}>
            <input
              required
              id='action'
              placeholder='What do you do now?'
              autocomplete='off'
              type='text'
              value={input()}
              onChange={e => setInput(e.currentTarget.value)}
            />
            <button type='submit'>go</button>
          </form>
        </div>
      </>
    )
  }
}

export default view
