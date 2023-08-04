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
    let completionEl;
    let promptEl;

    return (partial: any) => {
      if (partial.type === 'template') {
        staticPrompt = partial.system + (partial.preprompt ? `\n${partial.preprompt}` : '')

        render(
          root,
          `
            <div id='prompt'><p>${staticPrompt}</p></div>
            <pre id='completion'><code></code></pre>
            <div id='controls'><button id='cancel'>cancel</button></div>
            <details><summary>template</summary>
              <pre><code>${partial.content}</code></pre>
            </details>
           `
        )

        const cancelButton = document.getElementById('cancel')!
        cancelButton.onclick = () => {
          template.model.cancel()
          renderRedoButton()
        }
        cancelButton.style.display = 'block'
      } else {
        completionEl ??= document.getElementById('completion')
        promptEl ??= document.getElementById('prompt')

        const el = document.createElement('span')
        el.textContent = partial.content
        el.className = partial.type
        completionEl.appendChild(el)

        if (partial.type === 'gen') {
          render(promptEl, `<p>${staticPrompt}${"\nGenerate " + partial.prompt}</p>`)
        }
      }
    }
  }

  const template = await createTemplateCompletion()

  const completionResult = await template.collect(renderPartial(template))

  renderRedoButton()

  console.log(completionResult)
  console.log(JSON.parse(completionResult))
}
