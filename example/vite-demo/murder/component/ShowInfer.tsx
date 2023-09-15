import '../../style.css'
import { Template, StreamPartial } from 'ad-llama'
import { Accessor, For, Show, createEffect, createSignal } from 'solid-js'

export default function ShowInfer({ template, onComplete }: {
  template: Accessor<Template>,
  onComplete: (result: Awaited<ReturnType<Template['collect_refs']>>) => void
}) {
  const [canCancel, setCanCancel] = createSignal(true)
  const [prompt, setPrompt] = createSignal('')
  const [templateText, setTemplateText] = createSignal('')
  const [partials, setPartials] = createSignal<StreamPartial[]>([])
  const [error, setError] = createSignal<any>()

  createEffect(() => {
    setPartials([])

    template().collect_refs(partial => {
      switch (partial.type) {
        case 'ungen':
          setPartials([...partials().slice(0, -partial.tokenCount)])
          break
        case 'lit':
        case 'gen':
          setPartials([...partials(), partial])

          if (partial.type === 'gen') {
            setPrompt(partial.prompt)
          }

          break
        case 'template':
          setTemplateText(partial.content)
          break
      }
    })
      .then(results => {
        setCanCancel(false)
        onComplete(results)
      })
      .catch(error => {
        setError(error)
      })
  })

  return (
    <div>
      <details><summary>template</summary>
        <pre><code>{templateText()}</code></pre>
      </details>
      <div id='prompt'><p>{prompt()}</p></div>
      <pre id='completion'><code>
        <For each={partials()}>{(partial) =>
          <span classList={{ [partial.type]: true }}>{partial.content}</span>
        }</For>
      </code></pre>
      <Show when={canCancel()}>
        <div id='controls'>
          <button onClick={() => template().model.cancel()}>cancel</button>
        </div>
      </Show>
      <Show when={error()}>
        <div style={{color: 'red'}}>{error().toString()}</div>
      </Show>
    </div>
  )
}
