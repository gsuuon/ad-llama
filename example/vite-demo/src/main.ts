import './style.css'
import { ad, guessModelSpecFromPrebuiltId, loadModel } from 'ad-llama'

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div>
    <h1>
      <a href='https://github.com/gsuuon/ad-llama'><span id='gh'></span></a>
      ad-llama
    </h1>
    <p id='system'></p>
    <span id='preprompt'></span><span id='prompt'></span>
    <pre>
      <code id='completion'>
      </code>
    </pre>
    <div id='redo'></div>
    <pre>
      <code id='template'>
      </code>
    </pre>
    <pre><code id='progress'></code></pre>
  </div>
`

if (import.meta.hot) { import.meta.hot.accept() }

const progressEl = document.querySelector('#progress')!

const gen = ad(
  await loadModel(
    guessModelSpecFromPrebuiltId('Llama-2-7b-chat-hf-q4f32_1'),
    undefined,
    report => {
      progressEl.textContent = JSON.stringify(report, null, 2)
    }
  )
)

const generate = async () => {
  const { template, a } = gen('You are a dungeon master.', 'Create an interesting character based on the Dungeons and Dragons universe.')

  const result = template`
{
  "description": "${(a('description', {maxTokens: 1000, stops: ['\n']}))}",
  "name": "${(a('name'))}",
  "weapon": "${a('weapon')}",
  "items": [
    {
      "name": "${a('name')}",
      "description": "${a('short description')}",
      "type": "${a('type')}"
    },
    {
      "name": "${a('name')}",
      "description": "${a('short description')}",
      "type": "${a('type')}"
    },
    {
      "name": "${a('name')}",
      "description": "${a('short description')}",
      "type": "${a('type')}"
    }
  ]
}
`

  const completionEl = document.querySelector('#completion')!
  const promptEl = document.querySelector('#prompt')!
  const redoEl = document.querySelector('#redo')!

  const text = await result.collect(partial => {
    if (partial.type === 'template') {
      document.querySelector('#template')!.textContent = partial.content
      document.querySelector('#system')!.textContent = partial.system
      document.querySelector('#preprompt')!.textContent = partial.preprompt ?? ''
    } else {
      const el = document.createElement('span')
      el.textContent = partial.content
      el.className = partial.type
      completionEl.appendChild(el)

      if (partial.type === 'gen') {
        promptEl.textContent = " Generate " + partial.prompt
      }
    }
  })

  if (!redoEl.firstChild) {
    const button = document.createElement('button')
    button.onclick = () => {
      completionEl.textContent = ''
      promptEl.textContent = ''
      generate()
    }
    button.textContent = 'redo'
    redoEl.appendChild(button)
  }

  console.log(text)
  console.log(JSON.parse(text))
}

generate()
