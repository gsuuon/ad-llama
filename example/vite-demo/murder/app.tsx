import { createSignal } from 'solid-js'
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
  initialAppModel: AppModel
}) => {
  const [appModel, setAppModel] = createSignal<AppModel>(initialAppModel)

  const createTemplate = ad(llm)

  return (
    // Using Dynamic here doesn't re-render consistently - I could carefully use signals in every view but
    // I think the expectation would be that an update triggers a re-render
    <>
      {show(view, appModel().state, {
          update: setAppModel,
          llm,
          model: appModel,
          ...createTemplate
        })
      }
    </>
  )
}

export default App
