import { render } from 'solid-js/web'
import { LoadedModel, ad } from 'ad-llama'

import ShowInfer from './component/ShowInfer'
import Loading from './component/Loading'
import { Show, createSignal } from 'solid-js'

const app = document.getElementById('app')!

const TestInfer = ({ model }: { model: LoadedModel }) => {
  const { context, prompt } = ad(model)

  const assistant = context('You are a helpful assistant.', 'count to ten', { temperature: 0.1 })

  return (
    <ShowInfer
      template={assistant`one, two, three, ${prompt('')}, five, ${prompt('')}`}
      onComplete={results => console.log(results)}
    />
  )
}

const App = () => {
  const [model, setModel] = createSignal<LoadedModel | undefined>()

  return (
    <Show
      when={model() !== undefined}
      fallback={
        <Loading
          llamaModel='Llama-2-7b-chat-hf-q4f32_1'
          onLoad={setModel}
        />
      }>
      <TestInfer model={model()!} />
    </Show>
  )
}

render(() => <App />, app)
