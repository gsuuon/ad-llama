import config from './rollup.config.mjs'
import serve from 'rollup-plugin-serve'

export default {
  ...config,
  output: {
    ...config.output,
    file: 'public/index.js'
  },
  plugins: [
    ...config.plugins,
    serve('public')
  ]
}
