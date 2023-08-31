import { LoadReport, LoadedModel, loadModel} from 'ad-llama'
import { Show, batch, createEffect, createSignal } from 'solid-js'

export default function Loading({
  llamaModel,
  onLoad
}: {
  llamaModel: string,
  onLoad: (model: LoadedModel) => any
}) {
  const [report, setReport] = createSignal<LoadReport>({})
  const [loadPercent, setLoadPercent] = createSignal(0)
  const [loadSource, setLoadSource] = createSignal('')
  const [loadGpuShaders, setLoadGpuShaders] = createSignal<string>('')
  const [device, setDevice] = createSignal('')
  const [ready, setReady] = createSignal(false)
  const [error, setError] = createSignal<string | undefined>(undefined)

  loadModel(llamaModel, setReport).then(x => {
    setReady(true)
    onLoad(x)
  }).catch(e => {
    setError(e.message)
  })

  createEffect(() => {
    const report_ = report()

    batch( () => {
      if (report_.error) {
        setError(report_.error)
      }

      if (report_.loadModelFromWeb) {
        setLoadPercent(report_.loadModelFromWeb * 100)
        setLoadSource('web')
      } else if (report_.loadModelFromCache) {
        setLoadPercent(report_.loadModelFromCache * 100)
        setLoadSource('cache')
      }

      if (report_.loadGPUShaders) {
        if (typeof report_.loadGPUShaders === 'number') {
          setLoadGpuShaders((report_.loadGPUShaders * 100).toFixed(2) + '%')
        } else {
          setLoadGpuShaders(report_.loadGPUShaders)
        }
      }

      if (report_.targetDevice) {
        if (report_.targetDevice === 'gpu' && report_.detectGPU) {
          setDevice(`GPU (${report_.detectGPU})`)
        } else {
          setDevice(report_.targetDevice)
        }
      }
    })
  })

  return (
    <div>
      <div>Model: {llamaModel}</div>
      <Show when={error() !== undefined}>
        <div class='error'>Error: {error()}</div>
      </Show>
      <Show when={device() !== ''}>
        <div>Device: {device()}</div>
      </Show>
      <Show when={loadSource() !== ''}>
        <div>Source: {loadSource()}</div>
        <Show when={loadPercent() !== 0}>
          <div>{loadPercent().toFixed(2)}%</div>
        </Show>
      </Show>
      <Show when={loadGpuShaders() !== ''}>
        <div>Loading GPU shaders: {loadGpuShaders()}</div>
      </Show>
      <Show
        when={ready()}
        fallback={<h2>Loading..</h2>}
      >
        <h2>Ready</h2>
      </Show>
    </div>
  )
}
