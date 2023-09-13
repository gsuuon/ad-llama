import { createSignal, Show } from 'solid-js'
import { render } from 'solid-js/web'
import { LoadedModel } from 'ad-llama'

import Loading from './component/Loading'
import App from './app'


const Load = () => {
  const [model, setModel] = createSignal<LoadedModel | undefined>()

  return (
    <Show
      when={model()}
      fallback={
        <Loading
          llamaModel='Llama-2-7b-chat-hf-q4f32_1'
          onLoad={setModel}
        />
      }>{
        model => <App model={model()} />
    }</Show>
  )
}

render(() => <Load />, document.getElementById('app')!)
