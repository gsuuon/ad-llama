import { createSignal } from 'solid-js'
import { sample } from 'ad-llama'

import { Model } from '../model'
import { breakParagraph } from '../llm/util'
import ShowInfer from '../component/ShowInfer'

import { View } from './type'
import conversation from './conversation'
import scene from './scene'

const view: View<Model> = {
  ...conversation,
  ...scene,
  background: ({update, llm, context, a}) => {
    const { bias } = llm
    const { consistsOf } = sample

    const [template] = createSignal(
      // NOTE numCharacters here is just to guide the next characters field
      context('You are a text RPG game runner for a mystery game.')`{
        "setting": "${a('description of the world in which this game is set', {
          sampler: bias.reject(consistsOf([' Ravenswood', 'Ravenswood', 'Ravns Wood'])),
          maxTokens: 750,
          validate: {
            check: x => { return x.length > 50 },
            retries: 3,
            transform: x => x.replace(/\n/g, '\\n')
          }, // sometimes we just get 'Ravenswood'
        })}",
        "numCharacters": 3,
        "characters": ["${a('list of the full (first and last) names of the non-player characters', {
          sampler: bias.reject(consistsOf(['\n', '{'])),
        })}]
       }`
    )

    return <ShowInfer
      template={template}
      onComplete={
        result => {
          const background: {
            setting: string
            characters: string[]
          } = JSON.parse(result.completion)

          const [next, ...rest] = background.characters

          console.log({next, rest, background})

          update({
            state: 'character',
            background: {
              setting: background.setting,
              characterNames: background.characters
            },
            characters: [],
            currentCharacterName: next,
            pendingCharacterNames: rest
          })
        }
      }
    />
  },
  character: ({update, llm, context, a, model}) => {
    const { background, characters, currentCharacterName, pendingCharacterNames } = model()

    const characterGen = context(
      'You are a text RPG game runner for a murder-mystery game.',
      'Setting:\n'
        + background.setting
        + '\nNon-player characters:\n'
        + background.characterNames.map(x => ' - ' + x).join('\n')
        + '\n\n'
    )

    const [template] = createSignal(characterGen`{
      "name": "${currentCharacterName}",
      "role": "${a('role for this non-player character - what is their place in this story?')}",
      "description": "${a('description of this non-player character', {
      maxTokens: 600,
      ...breakParagraph(llm)
      })}",
      "secret": "${a('hidden secret about this character', breakParagraph(llm))}",
      "motivation": "${a('motivation of this character which may hint at their secret', breakParagraph(llm))}",
      "summary": "${a('one sentence summary of the character containing only publically known information about them', {stops:['.']})}"
    }`)

    return <ShowInfer
      template={template}
      onComplete={
        result => {
          const character: {
            name: string
            role: string
            description: string
            secret: string
            motivation: string
            summary: string
          } = JSON.parse(result.completion)

          const [next, ...rest] = pendingCharacterNames

          console.log({next, rest, character})

          if (next) {
            update({
              state: 'character',
              background: background,
              characters: [...characters, character],
              currentCharacterName: next,
              pendingCharacterNames: rest
            })
          } else {
            update({
              state: 'scene generate',
              background,
              characters,
              scenes: []
            })
          }
        }
      }
    />
  }
}

export default view
