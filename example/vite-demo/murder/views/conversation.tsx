import { View } from './type'
import { ModelConversation } from '../model'


const view: View<ModelConversation> = {
  "conversation parse input": () => <div>conversation parse input</div>,
  "conversation response": () => <div>conversation response</div>,
  conversation: () => <div>conversation</div>
}

export default view
