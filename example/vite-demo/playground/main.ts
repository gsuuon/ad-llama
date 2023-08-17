import './style.css'
import 'ad-llama' // prevent rollup treeshake
import * as adLlama from 'ad-llama'
import { renderTemplate } from '../renderTemplate'
import { basicSetup, EditorView } from 'codemirror'
import { Compartment }  from '@codemirror/state'
import { javascript } from '@codemirror/lang-javascript'
import placeholder from './placeholder'

  // @ts-expect-error: unused here but available in eval
const { ad, TargetDevice, loadModel, sample } = adLlama

if (import.meta.hot) { import.meta.hot.accept() }

const inference = document.getElementById('inference') as HTMLDivElement
const editor = document.getElementById('editor') as HTMLDivElement
const submit = document.getElementById('submit') as HTMLButtonElement
const editorError = document.getElementById('editor-error') as HTMLDivElement
submit.disabled = true

const language = new Compartment

editor.innerHTML = ''
const view = new EditorView({
  doc: placeholder,
  extensions: [
    basicSetup,
    EditorView.lineWrapping,
    language.of(javascript()),
  ],
  parent: editor
})

const editorContainer = document.querySelector('div.cm-editor') as HTMLDivElement
console.log({editorContainer})

const onresize = new ResizeObserver( () => {
  console.log('resize')
  view.requestMeasure()
})

onresize.observe(editorContainer)

const model =
  await loadModel(
    'Llama-2-7b-chat-hf-q4f32_1',
    report => inference.innerHTML = `<pre id='progress'><code>${JSON.stringify(report, null, 2)}</code></pre>`,
    new URLSearchParams(window.location.search).get('cpu') === null
      ? TargetDevice.GPU
      : TargetDevice.CPU 
  )

const run = async (code: string) => {
  editorError.innerHTML = ''
  await model.cancel()

  // @ts-expect-error: unused here but available in eval
  const createCtx = ad(model)

  renderTemplate(inference, async () => {
    try {
      return eval(code)
    } catch (e) {
      if (e instanceof Error) {
        editorError.innerText = e.message
      }
      throw e
    }
  })
}

submit.onclick = () => run(view.state.doc.toString())
submit.disabled = false
