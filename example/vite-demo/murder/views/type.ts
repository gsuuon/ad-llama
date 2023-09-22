import { Accessor, JSX } from 'solid-js'
import { CreateTemplate, LoadedModel } from 'ad-llama'

// TODO in/out is a thing now
// https://devblogs.microsoft.com/typescript/announcing-typescript-4-7/#optional-variance-annotations-for-type-parameters
export type View<AppModel extends { state: string }, SubModel extends AppModel = AppModel> = {
  [key in SubModel['state']]: (
    props:
      { model: Accessor<Extract<SubModel, { state: key }>> }
        & CreateTemplate
        & { llm: LoadedModel }
        & { update: (model: SubModel) => void }
        & { update: (model: AppModel) => void }
  ) => JSX.Element
}
