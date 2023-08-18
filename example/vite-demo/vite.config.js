import { resolve } from 'path'

export default {
  build: {
    target: 'esnext',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        hn: resolve(__dirname, 'hn/index.html'),
        murder: resolve(__dirname, 'murder/index.html'),
        playground: resolve(__dirname, 'playground/index.html')
      }
    }
  }
}
