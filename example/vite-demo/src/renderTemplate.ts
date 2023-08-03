const render = (el: HTMLElement, html: string) => el.innerHTML = html

export const renderTemplate = async (root: HTMLElement, createTemplateCompletion: any) => {
  const renderPartial = () => {
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
            <details><summary>template</summary>
              <pre><code>${partial.content}</code></pre>
            </details>
           `
        )
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

  const completionResult = await (await createTemplateCompletion()).collect(renderPartial())

  const button = document.createElement('button')
  button.onclick = () => renderTemplate(root, createTemplateCompletion)
  button.textContent = 'redo'
  button.style.display = 'block'
  document.getElementById('completion')?.appendChild(button)

  console.log(completionResult)
  console.log(JSON.parse(completionResult))
  console.dir(JSON.parse(completionResult))
}
