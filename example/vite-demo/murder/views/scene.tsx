import { For, createSignal } from 'solid-js'
import { sample } from 'ad-llama'

import { Model, ModelScene } from '../model'
import ShowInfer from '../component/ShowInfer'
import { breakParagraph } from '../llm/util'

import { View } from './type'

import './scene.css'


const view: View<Model, ModelScene> = {
  'scene generate': ({ update, model, llm, context, a }) => {
    const { background, characters, scenes, playerSceneInput } = model()

    const gameRunner = context(
      'You are a text RPG game runner for a murder-mystery game.',
      'Setting:\n'
      + background.setting
      + '\nNon-player characters:\n'
      + characters.map(char => char.summary).join('\n - ')
      + ( (scenes.length > 0) ? ('\nPrevious scene:\n' + scenes[scenes.length - 1].description) : '' )
      + ( playerSceneInput ?
          ( '\n\nPlayer input:\n' + playerSceneInput
            + '\n\nDescribe the scene where the player is now, after their input happens'
          )
          : '\nSet the scene for where the player currently is.')
        )

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
  'scene player input': ({update, model}) => {
    const [input, setInput] = createSignal('')
    const { background, scene, scenes, characters } = model()

    return (
      <>
        <div>
          <h3>Setting</h3>
          <p>{background.setting}</p>
          <h3>Scene</h3>
          <p>{scene.description}</p>
          <h3>Characters present</h3>
          <ul>
            <For each={scene.characterNames}>
              {name => <li>{name}</li>}
            </For>
          </ul>
        </div>
        <div>
          <form onSubmit={e => {
            e.preventDefault()

            update({
              state: 'scene parse input',
              characters,
              background,
              scene,
              scenes,
              playerInput: input()
            })
          }}>
            <input
              required
              id='action'
              placeholder='What do you do now?'
              autocomplete='off'
              type='text'
              value={input()}
              onChange={e => setInput(e.currentTarget.value)}
            />
            <button type='submit'>make it so</button>
          </form>
        </div>
      </>
    )
  },
  'scene parse input': ({update, model, context, llm, the }) => {
    const { playerInput, scene, background, characters, scenes } = model()

    const inputTypes = {
      // Not sure there's a meaningful difference between interact and inspect
      inspect: 'The player is inspecting an object in the scene. This means they are simply looking at something in the scene, expecting more details. Their character is not touching anything in the current scene.',
      interact: 'The player is interacting with an object in the scene. If they attempt to open a door, or move a table, or if they are searching something that would require their character to physically touch anything.',
      travel: 'The player is traveling to another location besides the current scene. This means they are leaving the current scene location and attempting to head somewhere else.',
      talk: 'The player is starting or continuing a conversation with someone. If the player talks to anyone, it is considered a "talk" input.'
    }

    const categorizer = context(
      'You are an input categorizer for a text rpg game.',
      '\nThe current scene:\n'
      + scene.description
      + '\n\nCategorize the player input according to the following types:\n'
      + Object.entries(inputTypes).map( ([k,v]) => `"${k}" - ${v}`).join('\n')
      + '\n\nIf the player talks to someone then it is always a talk type.'
      + '\n\nThe player input: ' + playerInput
    )

    const [template] = createSignal(categorizer`{
  "type": "${the('type of the given player input', {
          sampler: llm.bias.accept(sample.oneOf(Object.keys(inputTypes))),
          temperature: 0.3
        })}"
}`)

    return <ShowInfer
      template={template}
      onComplete={
        result => {
          const input: {
            type: keyof typeof inputTypes
          } = JSON.parse(result.completion)

          console.log({input})

          switch (input.type) {
            case 'talk':
              update({
                state: 'conversation start',
                playerInput,
                background,
                scenes,
                characters,
                scene
              })
              break
            case 'travel':
              update({
                state: 'scene generate',
                background,
                scenes,
                characters,
                playerSceneInput: playerInput
              })
              break
            case 'inspect':
              update({
                state: 'scene update',
                background,
                scene,
                scenes,
                characters,
                playerSceneInput: playerInput
              })
              break
            case 'interact':
              console.log('regenerate scene with additional info, and maybe npc comments')
              break
          }
        }
      }
    />
  },
  'scene update': ({update, model, context, a}) => {
    const { scene, background, playerSceneInput, scenes, characters } = model()

    const gameRunner = context(
      'You are a text RPG game runner for a murder-mystery game.',
      'Setting:\n'
      + background.setting
      + '\n\nCurrent scene:\n'
      + scene.description
      + '\n\nPlayer input:\n'
      + playerSceneInput
    )

    const [template] = createSignal(gameRunner`{
      "description": "${a('description of the effect of the players input on the scene. If only additional details are revealed, write those additional details as if they followed organically from the original scene description.',
      {
        validate: {
          transform: x => x.replace(/\n/g, '\\n')
        }
      })}"
    }`)

    return <ShowInfer
      template={template}
      onComplete={
        result => {
          const sceneAdditional: {
            description: string
          } = JSON.parse(result.completion)

          const updatedScene = {
            ...scene,
            description: scene.description + '\n\n' + sceneAdditional.description
          }

          console.log({sceneAdditional})

          update({
            state: 'scene player input',
            scene: updatedScene,
            scenes: [...scenes.slice(0, -1), updatedScene],
            background,
            characters
          })
        }
      }
    />
  }
}

export default view
