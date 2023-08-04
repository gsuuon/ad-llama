import { resolve } from 'path'

export default {
  build: {
    target: 'esnext',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        hn: resolve(__dirname, 'src/hn/index.html'),
      }
    }
  }
}
