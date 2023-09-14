import { View } from './type'
import { ModelScene } from '../model'


const view: View<ModelScene> = {
  "scene generate": () => <div>scene generate</div>,
  "scene player input": () => <div>scene player input</div>
}

export default view
