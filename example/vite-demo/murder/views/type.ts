import { Accessor, JSX } from 'solid-js'
import { CreateTemplate, LoadedModel } from 'ad-llama'

export type View<AppModel extends { state: string }> = {
  [key in AppModel['state']]: (
    props:
      { model: Accessor<Extract<AppModel, { state: key }>> }
        & CreateTemplate
        & { llm: LoadedModel }
        & { update: (model: AppModel) => void }
  ) => JSX.Element
}
