import { resolve } from 'path'
import { defineConfig } from 'vite'
import solidPlugin from 'vite-plugin-solid'

export default defineConfig({
  build: {
    target: 'esnext',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        hn: resolve(__dirname, 'hn/index.html'),
        murder: resolve(__dirname, 'murder/index.html'),
        colorgen: resolve(__dirname, 'colorgen/index.html'),
        playground: resolve(__dirname, 'playground/index.html')
      },
      plugins: [
        {
          name: 'disable-treeshake-playground',
          transform(code, id) {
            if (/playground\/main.ts/.test(id)) {
              return {
                code,
                map: null,
                moduleSideEffects: 'no-treeshake'
              }
            }
            return null
          }
        }
      ]
    }
  },
  plugins: [solidPlugin()]
})
