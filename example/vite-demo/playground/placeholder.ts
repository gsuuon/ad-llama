export default `const { template, a, __ } = createCtx(
  'You are a dungeon master.',
  'Create an interesting character based on the Dungeons and Dragons universe.'
)

const { bias } = model
const { oneOf, consistsOf, chars } = sample

template\`{
  "class": "\${a('main class')}",
  "subclass": "\${a('sub class')}",
  "name": "\${(a('name'))}",
  "weapon": "\${a('special weapon')}",
  "description": "\${(a('clever description', {
    maxTokens: 1000,
    stops: ['\\n'],
    validate: {
      check: x => x.length > 10,
      retries: 2
    }
  }))}",
  "heightInCm": \${a('height in cm', {
    sampler: bias.accept(chars.number)
  })},
  "appearance": "\${a('description of their appearance')}",
  "age": \${__('Generate an age', {
    sampler: bias.accept(chars.number),
    maxTokens: 3
  })},
  "items": [
    {
      "name": "\${a('name')}",
      "description": "\${a('short description')}",
      "type": "\${a('type')}"
    }
  ]
}\``
