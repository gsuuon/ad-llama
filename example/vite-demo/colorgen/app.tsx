import { For, JSX, Show, createSignal } from 'solid-js'
import { Dynamic, render } from 'solid-js/web'
import { CreateTemplate, LoadedModel, ad, sample } from 'ad-llama'

import ShowInfer from './components/ShowInfer'
import Loading from './components/Loading'

type AppModel = {
  state: 'primary' // generate a primary color
} | {
  state: 'hex' // generate a shade of that color as a hex
  primary: string
} | {
  state: 'show' // show the color
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
    const [showAgain, setShowAgain] = createSignal(true)

    return (
      <div style={`display: flex; flex-direction: column; align-items: center;`}>
        <h3>{color} is a shade of {primary}</h3>
        <div style={`width: ${size}; height: ${size}; background-color: #${color}`}>
        </div>
        <Show when={showAgain()} >
          <button style={`margin: 2rem`} onClick={() => {
            setShowAgain(false)
            update({state: 'primary'})
          }}>again</button>
        </Show>
      </div>
    )
  },
  hex: ({ context, update, primary, a }) => {
    const assistant = context('You are a helpful coloring assistant.')

    const [hexColor] = createSignal(
      assistant`#${a(`shade of ${primary} as a hex rgb value`, {
        stops: ['\n', ' '],
        maxTokens: 6,
        validate: {
          retries: 10,
          check: x => x.length === 6
        },
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
      assistant`Color: ${prompt('pick a primary color between red, green or blue', {
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
  const [appModels, setAppModels] = createSignal<AppModel[]>([{state: 'primary'}])

  const createTemplate = ad(model)

  const addAppModel = (appModel: AppModel) => setAppModels([...appModels(), appModel])

  return (
    <For each={appModels()}>{appModel =>
      <Dynamic
        component={view[appModel.state]}
        update={addAppModel}
        model={model}
        {...appModel}
        {...createTemplate}
      />
    }</For>
  )
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
