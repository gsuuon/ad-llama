export default `const { template, a, __ } = createCtx(
  'You are a dungeon master.',
  'Create an interesting non-player character based on the Dungeons and Dragons universe.'
)

const classes = [ 'Barbarian', 'Bard', 'Cleric', 'Druid', 'Fighter', 'Monk', 'Paladin', 'Ranger', 'Rogue', 'Sorcerer', 'Warlock', 'Wizard' ]

const { bias } = model
const { oneOf, consistsOf, chars } = sample

template\`{
  "class": "\${a('main class', { sampler: bias.accept(oneOf(classes)) })}",
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
  "appearance": "\${a('description of the character\\'s appearance')}",
  "age": \${a('fitting age', {
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
