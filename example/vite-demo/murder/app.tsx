import { ValidComponent, createSignal } from 'solid-js'
import { Dynamic } from 'solid-js/web'
import { LoadedModel, ad } from 'ad-llama'

import { View } from './views/type'

const App = <Model extends {state: string}>({
  model,
  view,
  initialAppModel
}: {
  model: LoadedModel
  view: View<Model>
  initialAppModel: Model
}) => {
  const [appModel, setAppModel] = createSignal<Model>(initialAppModel)

  const createTemplate = ad(model)

  return <Dynamic
    component={view[appModel().state as Model['state']] as ValidComponent}
    update={setAppModel}
    model={model}
    {...appModel()}
    {...createTemplate}
  />
}

export default App
