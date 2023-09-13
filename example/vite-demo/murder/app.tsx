import { createSignal } from 'solid-js'
import { Dynamic } from 'solid-js/web'
import { LoadedModel } from 'ad-llama'

const app = document.getElementById('app')!

const App = <AppModel,>({
  model,
}: {
  model: LoadedModel
}) => {
  const [appModel, setAppModel] = createSignal<AppModel>({state: 'primary'})

  const createTemplate = ad(model)

  return <Dynamic
    component={view[appModel().state]}
    update={setAppModel}
    model={model}
    {...appModel()}
    {...createTemplate}
  />
}

export default App
