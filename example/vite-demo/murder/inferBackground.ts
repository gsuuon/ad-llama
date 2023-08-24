import { sample, ad, LoadedModel } from 'ad-llama'
import { renderTemplate } from '../renderTemplate'

const { consistsOf } = sample

const infer = document.querySelector<HTMLDivElement>('#infer')!
const game = document.querySelector<HTMLDivElement>('#game')!
const render = (el: HTMLElement, html: string) => el.innerHTML = html

export const establishBackgroundAndCharacters = async (model: LoadedModel) => {
  const { bias } = model
  const { context, a } = ad(model)

  const breakParagraph = {
    // try to get a '\n' to stop at a paragraph
    // TODO sampler which starts preferring '\n' after a certain amount of tokens
    // for now we just retry if we got nothing
    sampler: bias.prefer(consistsOf(['\n']), 1.2),
    stops: ['\n'],
    validate: { check: (x: string) => x.length > 1, retries: 3 }
  }

  const text = (body: string, className?: string) => {
    const el = document.createElement('p')
    if (className) {
      el.className = className
    }
    el.textContent = body
    return el
  }

  const background: {
    setting: string,
    numCharacters: number,
    characters: string[]
  } = JSON.parse(
    await renderTemplate(infer, async () =>
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
    , false)
  )

  const characterNames: string[] = background.characters
  console.log({characterNames})

  render(game, `
    <div id='scene'></div>
    <h2>Setting</h2>
    <p>${background.setting}</p>
    <h2>Characters</h2>
    <ul>
    ${characterNames.map(name => `<li id="${name}">${name}</li>`).join('')}
    </ul>`
  )

  let characters: {
    name: string
    role: string
    description: string
    secret: string
    motivation: string
    summary: string
  }[] = []

  for (const name of characterNames) {
    const gameWithSetting = context(
      'You are a text RPG game runner for a murder-mystery game.',
      'Setting:\n'
      + background.setting
      + '\nNon-player characters:\n'
      + characterNames.map(x => ' - ' + x).join('\n')
      + '\n\n'
    )

    const character = JSON.parse(await renderTemplate(infer, async () => gameWithSetting`{
      "name": "${name}", 
      "role": "${a('role for this non-player character - what is their place in this story?')}",
      "description": "${a('description of this non-player character', {
      maxTokens: 600,
      ...breakParagraph
      })}",
      "secret": "${a('hidden secret about this character', breakParagraph)}",
      "motivation": "${a('motivation of this character which may hint at their secret', breakParagraph)}",
      "summary": "${a('one sentence summary of the character containing only publically known information about them', {stops:['.']})}"
    }`, false))

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

  const scene: {
    description: string,
    characters: string[]
  } = JSON.parse(await renderTemplate(infer, async () => {
    const gameRunner = context(
      'You are a text RPG game runner for a murder-mystery game.',
      'Setting:\n'
      + background.setting
      + '\nNon-player characters:\n'
      + characters.map(char => char.summary).join('\n - ')
      + '\nSet the scene for where the player currently is:')

    return gameRunner`{
      "description": "${a('description of the current scene', breakParagraph)}",
      "characters": ["${a('list of the names of the non-player characters present which the player could talk to')}]
    }`
  }, false))

  render(document.getElementById('scene')!, `
    <div id='actions'></div>
    <h2>Scene</h2>
    <p>${scene.description}</p>
    <h4>People present:</h4>
    <ul>
    ${scene.characters.map((name: string) => `<li id="${name}">${name}</li>`).join('')}
    </ul>`
  )

  return {
    characters,
    background,
    scene
  }
}
