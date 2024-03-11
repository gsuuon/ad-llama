import { builtinModules } from 'module'

import { nodeResolve } from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import typescript from 'rollup-plugin-typescript2';

export default {
  input: 'src/index.ts',
  output: {
    file: 'dist/index.js',
    format: 'esm',
    name: 'ad-llama',
    exports: 'named',
  },
  plugins: [
    nodeResolve({ browser: true }),
    commonjs({ ignore: builtinModules }),
    typescript()
  ],
};
