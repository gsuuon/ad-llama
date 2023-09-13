export type Model = {
  state: 'generate background'
} | {
  state: 'generate scene'
} | {
  state: 'generate character'
} | {
  state: 'conversation'
}

export const initial: Model = {
  state: 'generate background'
}
