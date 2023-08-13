import { Template, StreamPartial } from 'ad-llama'

const render = (el: HTMLElement, html: string) => el.innerHTML = html

export const renderTemplate = async (root: HTMLElement, createTemplateCompletion: () => Promise<Template>) => {
  const renderRedoButton = () => {
    const redoButton = document.createElement('button')
    redoButton.onclick = () => renderTemplate(root, createTemplateCompletion)
    redoButton.textContent = 'redo'
    document.getElementById('controls')?.appendChild(redoButton)
  }

  const renderPartial = (template: Template) => {
    let staticPrompt = ''
    let completionEl : HTMLElement | null;
    let promptEl : HTMLElement | null;

    return (partial: StreamPartial) => {
      switch (partial.type) {
        case 'template':
          staticPrompt = partial.system + (partial.preprompt ? `\n${partial.preprompt}` : '')

          render(
            root,
            `<div id='prompt'><p>${staticPrompt}</p></div>
             <details><summary>template</summary>
             <pre><code>${partial.content}</code></pre>
             </details>
             <pre id='completion'><code></code></pre>
             <div id='controls'><button id='cancel'>cancel</button></div>`
          )

          const cancelButton = document.getElementById('cancel')!
          cancelButton.onclick = () => {
            template.model.cancel()
            cancelButton.remove()
            renderRedoButton()
          }
          cancelButton.style.display = 'block'

          completionEl = document.getElementById('completion')
          promptEl = document.getElementById('prompt')

          break
        case 'ungen':
          for (let i = 0; i < partial.tokenCount; i++) {
            completionEl?.lastChild?.remove()
          }
          break
        case 'gen':
        case 'lit':
          const el = document.createElement('span')
          el.textContent = partial.content
          el.className = partial.type
          completionEl?.appendChild(el)

          if (partial.type === 'gen') {
            render(promptEl!, `<p>${staticPrompt} ${partial.prompt}</p>`)
          }
          break
      }
    }
  }

  const template = await createTemplateCompletion()

  const completionResult = await template.collect_refs(renderPartial(template))

  renderRedoButton()
  document.getElementById('cancel')?.remove()

  console.log(completionResult)
  console.log(JSON.parse(completionResult.completion))

  return completionResult
}
