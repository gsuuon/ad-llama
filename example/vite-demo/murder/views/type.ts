import { JSX } from 'solid-js'
import { CreateTemplate, LoadedModel } from 'ad-llama'

export type View<AppModel extends { state: string }> = {
  [key in AppModel['state']]: (
    props:
      Extract<AppModel, { state: key }>
        & CreateTemplate
        & { model: LoadedModel }
        & { update: (model: AppModel) => void }
  ) => JSX.Element
}
