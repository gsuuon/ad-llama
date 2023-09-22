import { Show, createSignal } from 'solid-js'
import { LoadedModel, ad } from 'ad-llama'

import { View } from './views/type'

// type helper
const show = <M extends Object, K extends keyof M, P>(map: M, key: K, props: P) => map[key](props)

const App = <AppModel extends {state: string}>({
  llm,
  view,
  initialAppModel
}: {
  llm: LoadedModel
  view: View<AppModel>
  initialAppModel: AppModel | AppModel[]
}) => {
  const [history, setHistory] = createSignal<AppModel[]>(
    Array.isArray(initialAppModel) ? initialAppModel : [initialAppModel]
  )

  const createTemplate = ad(llm)
  ;(window as any).examine = () => console.log(history())

  const update = (m: AppModel) => {
    setHistory(history().concat([m]))
  }

  return (
    // Using Dynamic here doesn't re-render consistently - I could carefully use signals in every view but
    // I think the expectation would be that an update triggers a re-render
    <>
      <Show when={history().length > 1}>
        <button onClick={() => {
          llm.cancel()
          setHistory(history().slice(0, -1))
        }}>undo</button>
      </Show>
      {show(view, history()[history().length - 1].state, {
          update,
          llm,
          model: () => history()[history().length - 1],
          ...createTemplate
        })
      }
    </>
  )
}

export default App
