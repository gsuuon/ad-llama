const render = (el: HTMLElement, html: string) => el.innerHTML = html

export const renderTemplate = async (root: HTMLElement, createTemplateCompletion: any) => {
  const renderRedoButton = () => {
    const redoButton = document.createElement('button')
    redoButton.onclick = () => renderTemplate(root, createTemplateCompletion)
    redoButton.textContent = 'redo'
    document.getElementById('controls')?.appendChild(redoButton)
  }

  const renderPartial = (template: any) => {
    let staticPrompt = ''
    let completionEl : HTMLElement | null;
    let promptEl : HTMLElement | null;

    return (partial: any) => {
      if (partial.type === 'template') {
        staticPrompt = partial.system + (partial.preprompt ? `\n${partial.preprompt}` : '')

        render(
          root,
          `
            <div id='prompt'><p>${staticPrompt}</p></div>
            <details><summary>template</summary>
              <pre><code>${partial.content}</code></pre>
            </details>
            <pre id='completion'><code></code></pre>
            <div id='controls'><button id='cancel'>cancel</button></div>
           `
        )

        const cancelButton = document.getElementById('cancel')!
        cancelButton.onclick = () => {
          template.model.cancel()
          cancelButton.remove()
          renderRedoButton()
        }
        cancelButton.style.display = 'block'
      } else {
        completionEl ??= document.getElementById('completion')
        promptEl ??= document.getElementById('prompt')

        if (partial.type === 'ungen') {
          for (let i = 1; i < partial.tokenCount; i++) {
            completionEl?.lastChild?.remove()
          }
          return
        }

        const el = document.createElement('span')
        el.textContent = partial.content
        el.className = partial.type
        completionEl?.appendChild(el)

        if (partial.type === 'gen') {
          render(promptEl!, `<p>${staticPrompt} ${partial.prompt}</p>`)
        }
      }
    }
  }

  const template = await createTemplateCompletion()

  const completionResult = await template.collect(renderPartial(template))

  renderRedoButton()
  document.getElementById('cancel')?.remove()

  console.log(completionResult)
  console.log(JSON.parse(completionResult))
}
