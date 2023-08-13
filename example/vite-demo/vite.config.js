import { resolve } from 'path'

export default {
  build: {
    target: 'esnext',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        hn: resolve(__dirname, 'hn/index.html'),
        mystery: resolve(__dirname, 'mystery/index.html'),
      }
    }
  }
}
