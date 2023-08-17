import { Template, StreamPartial } from 'ad-llama'

const render = (el: HTMLElement, html: string) => el.innerHTML = html

export const renderTemplateRefs = async (
  root: HTMLElement,
  createTemplateCompletion: () => Promise<Template>,
  allowRedoCancel = true
) => {
  const renderRedoButton = () => {
    const redoButton = document.createElement('button')
    redoButton.onclick = () => renderTemplateRefs(root, createTemplateCompletion)
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
            `<details><summary>template</summary>
             <pre><code>${partial.content}</code></pre>
             </details>
             <div id='prompt'><p>${staticPrompt}</p></div>
             <pre id='completion'><code></code></pre>
             <div id='controls'></div>`
          )

          if (allowRedoCancel) {
            const cancelButton = document.createElement('button')
            cancelButton.onclick = () => {
              template.model.cancel()
              cancelButton.remove()
              renderRedoButton()
            }
            cancelButton.id = 'cancel'
            cancelButton.textContent = 'cancel'
            cancelButton.style.display = 'block'
            document.getElementById('controls')?.appendChild(cancelButton)
          }

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

  if (allowRedoCancel) {
    renderRedoButton()
    document.getElementById('cancel')?.remove()
  }

  console.log(completionResult)

  try { console.log(JSON.parse(completionResult.completion)) } catch { }

  return completionResult
}

export const renderTemplate = async (
  root: HTMLElement,
  createTemplateCompletion: () => Promise<Template>,
  allowRedoCancel = true
) => (await renderTemplateRefs(root, createTemplateCompletion, allowRedoCancel)).completion
