import '../src/style.css'
import { renderTemplate } from '../src/renderTemplate'
import { TargetDevice, ad, guessModelSpecFromPrebuiltId, loadModel } from 'ad-llama'

if (import.meta.hot) { import.meta.hot.accept() }

const app = document.querySelector<HTMLDivElement>('#app')!

const hnApiGetRandomWhosHiring = async (tries = 2) => {
  const whosHiringPostRes = await fetch('https://hacker-news.firebaseio.com/v0/item/36956867.json')
  const whosHiring = await whosHiringPostRes.json()
  const random = Math.floor(Math.random() * whosHiring.kids.length)

  const listingId = whosHiring.kids[random] // uh...

  const hnListingRes = await fetch(`https://hacker-news.firebaseio.com/v0/item/${listingId}.json`)
  const hnListing = await hnListingRes.json()

  const text = hnListing.text

  if (text === '[DEAD]' || text === '[FLAGGED]' && tries > 0) {
    console.log('Got listing:', text, 'trying again..', tries)
    return await hnApiGetRandomWhosHiring(tries - 1)
  }

  return text
}

renderTemplate(app, async () => {
  const gen = ad(
    await loadModel(
      guessModelSpecFromPrebuiltId('Llama-2-7b-chat-hf-q4f32_1'),
      report => app.innerHTML = `<pre id='progress'><code>${JSON.stringify(report, null, 2)}</code></pre>`,
      new URLSearchParams(window.location.search).get('cpu') === null
        ? TargetDevice.GPU
        : TargetDevice.CPU 
    )
  )

  const listing = await hnApiGetRandomWhosHiring()

  const { template, a, __ } = gen(
    'You are a helpful assistant that catalogues job listings. For any requests for which there is not sufficient information based on the listing itself, fill the field with only "NA".',
    `\nThe listing:\n"""\n${listing}\n"""\n`,
    {
      preword: 'What is'
    }
  )

  return template`{
  "company": {
    "name": "${a('name for the company')}",
    "description": "${a('description of the company and/or what it does. Fill with "NA" if no information is provided.')}",
    "sector": "${a('sector that this company operates in')}",
    "links": ["${a('list of additional links from the listing')}]
  },
  "role": {
    "primary": "${a('primary role')}",
    "additionalRoles": [${a('list of any additional roles mentioned, or ["NA"] if there are none', {
      stops: ["NA", "{"],
    })}]
  },
  "salary": {
    "currency": "${a('currency label')}",
    "info": "${'any additional info about the salary if available, or NA if none'}",
    "max": "${a('maximum salary amount based on the listing, or NA if not mentioned')}",
    "min": "${a('minimum salary amount based on the listing, or NA if not mentioned')}"
  },
  "skills": ["${a('list of skills')}],
  "remote": {
    "allowed": ${__('fill a javascript boolean, true if remote is allowed for this listing, else false. If the listing says ONSITE, that means remote is not allowed.', {
      stops: ['}',' ','\n'],
    })},
    "info": "${a('short description of their remote policy based on the listing or NA')}"
  }
}`
})



