import '../style.css'
import { renderTemplate } from '../renderTemplate'
import { TargetDevice, ad, loadModel, validate, sample, TemplateExpressionOptions } from 'ad-llama'

if (import.meta.hot) { import.meta.hot.accept() }

const app = document.querySelector<HTMLDivElement>('#app')!

const hnApiGetRandomWhosHiring = async (tries = 2): Promise<string> => {
  const whosHiringPostRes = await fetch('https://hacker-news.firebaseio.com/v0/item/36956867.json')
  const whosHiring = await whosHiringPostRes.json()
  const random = Math.floor(Math.random() * whosHiring.kids.length)

  const listingId = whosHiring.kids[random] // uh...

  const hnListingRes = await fetch(`https://hacker-news.firebaseio.com/v0/item/${listingId}.json`)
  const hnListing = await hnListingRes.json()

  const text = hnListing.text

  if (text === '[dead]' || text === '[flagged]' && tries > 0) {
    console.log('Got listing:', text, 'trying again..', tries)
    return await hnApiGetRandomWhosHiring(tries - 1)
  }

  console.log({
    link: `https://news.ycombinator.com/item?id=${hnListing.id}`,
    date: new Date(hnListing.time * 1000)
  })

  return text
}

const result = await renderTemplate(app, async () => {
  const model = await loadModel(
    'Llama-2-7b-chat-hf-q4f16_1',
    report => app.innerHTML = `<pre id='progress'><code>${JSON.stringify(report, null, 2)}</code></pre>`,
    new URLSearchParams(window.location.search).get('cpu') === null
      ? TargetDevice.GPU
      : TargetDevice.CPU 
  )

  const { context, the, a, prompt } = ad(model)
  const { bias } = model
  const { oneOf, chars } = sample

  const listing = await hnApiGetRandomWhosHiring()

  const assistant = context(
    'You are a helpful assistant that summarizes job listing into structured data. Fill the next field in the given JSON string based on following listing.',
    `\nThe listing:\n"""\n${listing}\n"""\n`,
    { temperature: 0.25 }
  )

  const asComp: TemplateExpressionOptions = {
    sampler: bias.accept(chars.number),
    temperature: 0.1
  }

  const asStringList: TemplateExpressionOptions = {
    sampler: bias.accept(oneOf(['"', ']'])), // start off with a quote or end immediately
    validate: {
      check: validate.json.list,
      retries: 3
    }
  }

  return assistant`{
  "company": {
    "name": "${the('name for the company')}",
    "description": "${the('description for the company')}",
    "sector": "${the('sector that this company operates in')}",
    "links": ["${the('list of additional links from the listing')}]
  },
  "role": {
    "primary": "${the('primary role')}",
    "additionalRoles": [${the('list of any additional roles the listing may be hiring for', asStringList)}]
  },
  "compensation": {
    "currency": "${the('currency label')}",
    "max": ${the('maximum compensation amount for the primary role. Based solely on the listing - fill with 0 if no info about the compensation amount was given.', asComp)},
    "min": ${the('minimum compensation amount for the primary role based on the listing, or 0 if no info', asComp)}
    "info": "${prompt('Any additional info about compensation, or "no information provided"')}",
  },
  "skills": [${a('list of specific skills mentioned in the listing', asStringList)}],
  "remote": {
    "allowed": ${prompt('true if remote is allowed for this listing, else false. If the listing says onsite, that means remote is not allowed. If it says hybrid, that means remote is allowed but with limits', {
      sampler: bias.accept(oneOf(['true', 'false']))
    })},
    "info": "${prompt('Summarize all information in the listing related to remote work. Include information about geography or timezone. Mention if the listing says the company is fully remote or remote first.')}"
  }
}`})

try { console.log(JSON.parse(result)) } catch { }
