type Background = {
  setting: string
  characterNames: string[]
}

type Character = {
  name: string
  role: string
  description: string
  secret: string
  motivation: string
  summary: string
}

type Message = {
  character: Character | 'player'
  content: string
}

type Conversation = {
  sceneSummary: string
  messages: Message[]
}

type Scene = {
  characterNames: string[]
  conversations: Conversation[]
  description: string
  summary?: string
}

export type ModelScene = {
  state: 'scene generate'
  background: Background
  characters: Character[]
  scenes: Scene[]
  playerSceneInput?: string // what the player said to trigger a scene generation
} | {
  state: 'scene player input'
  background: Background
  characters: Character[]
  scenes: Scene[]
  scene: Scene
} | {
  state: 'scene parse input'
  playerInput: string
  background: Background
  characters: Character[]
  scenes: Scene[]
  scene: Scene
} | {
  state: 'scene update'
  background: Background
  characters: Character[]
  scenes: Scene[]
  scene: Scene
  playerSceneInput: string
}

export type ModelConversation = {
  // We parse out the actual content of what the player said from something like "I walk over and say hi"
  state: 'conversation start' 
  playerInput: string
  scene: Scene
  background: Background
  characters: Character[]
  scenes: Scene[]
} | {
  state: 'conversation response generate'
  conversation: Conversation
  background: Background
  character: Character
  characters: Character[]
  scenes: Scene[]
} | {
  state: 'conversation player input'
  conversation: Conversation
  background: Background
  character: Character
  characters: Character[]
  scenes: Scene[]
}

export type Model = {
  state: 'background generate'
} | {
  state: 'character generate'
  background: Background
  characters: Character[]
  currentCharacterName: string
  pendingCharacterNames: string[]
} | ModelScene
  | ModelConversation

export const initial: Model = {
  state: 'background'
}
