This is a simple solid-js demo that shows how to hook up components and signals to the loading and template inferencing steps of ad-llama. [./components](./components) has ShowInfer and Loading, which are one way of displaying UI for loading and inferencing. The main [./app.tsx](./app.tsx) is an MVU app which adds to the view on each update so we can see our history of templates and inferences.

Each run loops over three steps:
  1. generate a primary color
  1. generate a shade of that color
  1. present it

The ShowInfer uses a Template signal + createEffect to consume it so it will re-run if the template is updated.
