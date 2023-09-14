import { View } from './type'
import { Model } from '../model'

import conversation from './conversation'
import scene from './scene'

const view: View<Model> = {
  ...conversation,
  ...scene,
  background: () => <div>background</div>,
  character: () => <div>character</div>
}

export default view
