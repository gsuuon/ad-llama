import { View } from './type'
import { Message, ModelConversation } from '../model'
import ShowInfer from '../component/ShowInfer'
import { sample } from 'ad-llama'
import { For, createSignal } from 'solid-js'

const view: View<ModelConversation> = {
  'conversation start': ({update, context, llm, the, model}) => {
    const { scene, background, scenes, characters } = model()

    const messageExtract = context(
      'You are a conversation assistant in a text-based game.',
      'Extract what is said by the player in this input and who it is directed towards.'
      + '\nCurrent characters in the scene: ' + scene.characterNames.join(', ')
    )

    const [template] = createSignal(messageExtract`{
  "targetCharacterName": "${the('character the player is attempting to talk to', {
  sampler: llm.bias.accept(sample.oneOf(scene.characterNames))
})}",
  "content": "${the('words that the player says to the npc')}"
}`)

    return (
      <>
        <ShowInfer
          template={template}
          onComplete={
            result => {
              console.log({message: result.completion})

              const parsedMessage: {
                targetCharacterName: string
                content: string
              } = JSON.parse(result.completion)

              const targetChar = characters.find(char => char.name.includes(parsedMessage.targetCharacterName))

              if (targetChar === undefined) {
                // I guess we would generate a new character?
                console.log({characters})
                throw Error('Target character not found: ' + parsedMessage.targetCharacterName)
              }

              const message: Message = {
                content: parsedMessage.content,
                character: 'player'
              }

              update({
                state: 'conversation response generate',
                background,
                scenes,
                character: targetChar,
                characters,
                conversation: {
                  messages: [ message ],
                  sceneSummary: scene.description // TODO actually summarize
                }
              })
            }
          }
        />
      </>
    )
  },
  'conversation response generate': ({update, model, context, a}) => {
    const { character, conversation } = model()

    const roleplay = context(
      'You are role-playing as a character in a text based game having a conversation with the player',
      'Your character: ' + character.name + ' - ' + character.role
      + '\n\n' + character.description
      + '\n\nTheir secret: ' + character.secret
      + '\n\nTheir motivation: ' + character.motivation
    )

    const messages = conversation.messages.map(message => `[${message.character}] ${message.content}`).join('\n\n')

    const [template] = createSignal(roleplay`${messages}\n\n[${character.name}] ${a('response from the character', {id: 'response'})}\n`)

    return <ShowInfer
      template={template}
      onComplete={
        result => {
          update({
            ...model(),
            state: 'conversation player input',
            conversation: {
              messages: [...conversation.messages, {
                character,
                content: result.refs.response
              }],
              sceneSummary: conversation.sceneSummary
            },
          })
        }
      }
    />
  },
  'conversation player input': ({update, model}) => {
    const { conversation } = model()
    const [input, setInput] = createSignal('')

    return (
      <>
        <div>
          <For each={conversation.messages}>{
            message => <div>
              <h3>{typeof message.character === 'string' ? message.character : message.character.name}</h3>
              <p>{message.content}</p>
            </div>
          }</For>
        </div>
        <form onSubmit={e => {
          e.preventDefault()

          update({
            ...model(),
            conversation: {
              ...conversation,
              messages: [...conversation.messages, {
                character: 'player',
                content: input()
              }]
            },
            state: 'conversation response generate',
          })
        }}>
          <input
            required
            id='action'
            placeholder='What do you say?'
            autocomplete='off'
            type='text'
            value={input()}
            onChange={e => setInput(e.currentTarget.value)}
          />
          <button type='submit'>make it so</button>
        </form>
      </>
    )
  }
}

export default view
