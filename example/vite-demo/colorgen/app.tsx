import { JSX, Show, createSignal } from 'solid-js'
import { Dynamic, render } from 'solid-js/web'
import { CreateTemplate, LoadedModel, ad, sample } from 'ad-llama'

import ShowInfer from './component/ShowInfer'
import Loading from './component/Loading'

type AppModel = {
  state: 'primary'
} | {
  state: 'hex'
  primary: string
} | {
  state: 'show'
  primary: string
  color: string
}

type View = {
  [key in AppModel['state']]: (
    props: Extract<AppModel, { state: key }> & { model: LoadedModel } & CreateTemplate & { update: (model: AppModel) => void }
  ) => JSX.Element
}

const view: View = {
  show: ({ color, primary, update }) => {
    const size = '10rem'

    return (
      <div style={`display: flex; flex-direction: column; align-items: center;`}>
        <h3>{color} is a shade of {primary}</h3>
        <div style={`width: ${size}; height: ${size}; background-color: #${color}`}>
        </div>
        <button style={`margin: 2rem`} onClick={() => update({state: 'primary'})}>redo</button>
      </div>
    )
  },
  hex: ({ context, update, primary, a }) => {
    const assistant = context('You are a helpful coloring assistant.')

    const [hexColor] = createSignal(
      assistant`#${a(`specific shade of ${primary} as a hex rgb value`, {
        stops: ['\n', ' '],
        maxTokens: 6,
        id: 'hex'
      })} `
    )

    return (
      <ShowInfer
        template={hexColor}
        onComplete={results => {
          update({
            state: 'show',
            color: results.refs.hex,
            primary
          })
        }}
      />
    )
  },
  primary: ({ model, context, prompt, update }) => {
    const assistant = context('You are a helpful coloring assistant.')

    const [pickPrimaryColor] = createSignal(
      assistant`Color: ${prompt('pick a primary color', {
        sampler: model.bias.accept(sample.oneOf(['red\n', 'green\n', 'blue\n'])),
        stops: ['\n'],
        id: 'color'
      })}`
    )

    return (
      <ShowInfer
        template={pickPrimaryColor}
        onComplete={results => {
          update({
            state: 'hex',
            primary: results.refs.color
          })
        }}
      />
    )
  }
}

const App = ({model}: {model: LoadedModel}) => {
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
      }
    </Show>
  )
}

render(() => <Load />, document.getElementById('app')!)
