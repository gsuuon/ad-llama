import { View } from './type'
import { Conversation, Message, Model, ModelConversation, Scene } from '../model'
import ShowInfer from '../component/ShowInfer'
import { sample } from 'ad-llama'
import { For, createSignal, onMount } from 'solid-js'

const showConversation = (conversation: Conversation) =>
  conversation.messages.map(message => `[${message.character.name}] ${message.content}`).join('\n\n')

const view: View<Model, ModelConversation> = {
  'conversation start': ({update, context, llm, the, model}) => {
    const { scene, background, scenes, characters, playerInput } = model()

    const messageExtract = context(
      'You are a conversation assistant in a text-based game.',
      'Extract what is said by the player in this input and who it is directed towards.'
      + '\nCurrent characters in the scene: ' + scene.characterNames.join(', ')
      + '\n\nPlayer input: ' + playerInput
      + '\n\n'
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
                character: {
                  name: 'player',
                  role: 'player'
                }
              }

              update({
                state: 'conversation response generate',
                background,
                scenes,
                characters,
                conversation: {
                  messages: [ message ],
                  sceneDescription: scene.description, // TODO actually summarize
                  character: targetChar,
                }
              })
            }
          }
        />
      </>
    )
  },
  'conversation response generate': ({update, model, context, a}) => {
    const { conversation, scenes } = model()
    const character = conversation.character

    const roleplay = context(
      'You are role-playing as a character in a text based game having a conversation with the player',
      'The current scene:\n' + scenes[scenes.length - 1].description
      + 'Your character: ' + character.name + ' - ' + character.role
      + '\n\n' + character.description
      + '\n\nTheir secret: ' + character.secret
      + '\n\nTheir motivation: ' + character.motivation
    )

    const messages = showConversation(conversation)

    const [template] = createSignal(roleplay`${messages}\n\n[${character.name}] ${a('response from the character', {id: 'response'})}\n`)

    return <ShowInfer
      template={template}
      onComplete={
        result => {
          update({
            ...model(),
            state: 'conversation player input',
            conversation: {
              ...conversation,
              messages: [...conversation.messages, {
                character,
                content: result.refs.response
              }],
            },
          })
        }
      }
    />
  },
  'conversation player input': ({update, model}) => {
    const { conversation, scenes } = model()
    const [input, setInput] = createSignal('')

    const scene = scenes[scenes.length - 1]

    let leaveButton: HTMLButtonElement | undefined;
    onMount(() => leaveButton?.scrollIntoView())

    return (
      <>
        <div>
          <h3>Scene</h3>
          <p>{scene.description}</p>
        </div>
        <div>
          <For each={conversation.messages}>{
            message => <div>
              <h3>{message.character.name}</h3>
              <p>{message.content}</p>
            </div>
          }</For>
        </div>
        <form onSubmit={e => {
          e.preventDefault()

          update({
            ...model(),
            state: 'conversation response generate',
            conversation: {
              ...conversation,
              messages: [...conversation.messages, {
                character: {
                  role: 'player',
                  name: 'player'
                },
                content: input()
              }]
            },
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
          <button type='submit'>say</button>
        </form>
        <div style={`display: flex;`}>
          <button
            style={`margin: 1rem auto;`}
            onClick={
              () => update({
                ...model(),
                state: 'conversation summarize',
              })
            }
            ref={leaveButton}
          >leave</button>
        </div>
      </>
    )
  },
  'conversation summarize': ({update, context, model, the}) => {
    const { conversation, scenes } = model()

    return <ShowInfer
      template={
        context(
          'You are a conversation summarizer of a text game. You very short and to-the-point summaries.',
          'The player had a conversation with a character:\n' + showConversation(conversation)
        )`{
          "summary": "${the('shortest summary of the conversation', { stops: ['\n'] })}"
        }`
      }

      onComplete={
        result => {
          const res: {
            summary: string
          } = JSON.parse(result.completion)

          // pretty dirty assumptions here but it'll work

          const conversation_: Conversation = {
            ...conversation,
            summary: res.summary
          }

          const currentScene = scenes[scenes.length - 1]

          const scene: Scene = {
            ...currentScene,
            description: currentScene.description.trimEnd() + `\n\nThe player spoke with ${conversation_.character.name}. ` + conversation_.summary,
            conversations: [...currentScene.conversations.slice(0, -1), conversation_]
          }

          update({
            ...model(),
            state: 'scene player input',
            scene,
            scenes: scenes.slice(0, -1).concat([scene])
          })
        }
      }
    />
  }
}

export default view
