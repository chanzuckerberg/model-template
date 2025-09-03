# VCP Model Template

This is a template for creating a new model for submission to the [Virtual Cells Platform](https://virtualcellmodels.cziscience.com/).

To use it, follow the instructions below:

1. Clone this repo: `git clone https://github.com/chanzuckerberg/model-template.git`
2. Install [copier](https://copier.readthedocs.io/). If you are using `uv`, you can install it with `uv tool install copier`.
3. Run `copier copy . <model-name>/` to generate your model under the `<model-name>` directory.

You will be prompted to answer a few questions, and based on your responses, the necessary files and directories will be created for your model.

The generated model directory will contain the following files and folders:

- `<model-name>_mlflow_pkg/`: The MLflow package for the model
- `model_card_docs/`: Metadata for the model, including the contents of the model card.
- `.copier-answers.yml`: Configuration file with answers to the prompts during model generation.
- `copier.yml`: Configuration file for the copier tool.
