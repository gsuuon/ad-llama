import '../src/style.css'
import { renderTemplate } from '../src/renderTemplate'
import { ad, guessModelSpecFromPrebuiltId, loadModel } from 'ad-llama'

if (import.meta.hot) { import.meta.hot.accept() }

const app = document.querySelector<HTMLDivElement>('#app')!

const hnApiGetRandomWhosHiring = async () => {
  const whosHiringPostRes = await fetch('https://hacker-news.firebaseio.com/v0/item/36956867.json')
  const whosHiring = await whosHiringPostRes.json()
  const random = Math.floor(Math.random() * whosHiring.kids.length)

  const listingId = whosHiring.kids[random] // uh...

  const hnListingRes = await fetch(`https://hacker-news.firebaseio.com/v0/item/${listingId}.json`)
  const hnListing = await hnListingRes.json()

  return hnListing.text
}

renderTemplate(app, async () => {
  const gen = ad(
    await loadModel(
      guessModelSpecFromPrebuiltId('Llama-2-7b-chat-hf-q4f32_1'),
      undefined,
      report => app.innerHTML = `<pre id='progress'><code>${JSON.stringify(report, null, 2)}</code></pre>`
    )
  )

  const listing = await hnApiGetRandomWhosHiring()

  const { template, a } = gen('You are a helpful assistant that catalogues job listings.', listing)

  return template`{
  "company": {
    "name": "${a('name')}",
    "description": "${a('description')}",
    "sector": "${a('sector that this company operates in')}",
    "links": ["${a('list of additional links from the listing')}]
  },
  "role": {
    "primary": "${a('role')}",
    "additionalRoles": [${a('list of any additional roles mentioned, or ["NA"] if there are none', { stops: ["NA"]})}]
  },
  "salary": {
    "currency": "${a('currency label')}",
    "info": "${'any additional info about the salary if available, or NA if none'}",
    "max": "${a('maximum salary amount based on the listing, or NA if not mentioned')}",
    "min": "${a('minimum salary amount based on the listing, or NA if not mentioned')}"
  },
  "skills": ["${a('list of skills')}],
  "remote": {
    "allowed": ${a('boolean, true if remote is allowed else false', { stops: ['}',' ','\n']})},
    "info": "${a('short description of their remote policy based on the listing or NA')}"
  }
}`
})



