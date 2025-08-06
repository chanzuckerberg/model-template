# Why use MLflow to package models

MLflow allows you to package and distribute your model in a way that makes it easy for other users to use your model to run inference. It also makes it easy for model serving platforms (ex: sagemaker, databricks, azureML, etc), to serve your model as a web service. Read about the [MLflow Model Format](https://mlflow.org/docs/latest/ml/model) to learn more.

Specifically, using MLflow provides the following concrete benefits:

1. Packages your model inference code and weights in a uniform way that is independent of the ML framework used to train the model. This significantly improves the distribution of your model.

2. Provides a schema specification of the input and output of the model which enables automatic input and output data validation at inference time. It also enables the generation of crystal clear documentation for your model.

3. Provides a uniform scoring API that is independent of the ML framework used to train the model. This significantly improves the usability of your model and deployability of your model in model serving platforms.

You can read about the [MLflow Model Format](https://mlflow.org/docs/latest/ml/model) to learn more. You an also see examples of **MLflow Model Artifacts** in the **Examples** directory. 

# Setup

1. Create and activate a python virtual environment **with the python version supported by your model. Your model must support `python>=3.10` for compatibility with other downstream systems**.

For virtual environment library, **you must use `conda` or `virtualenv`** to be compatible with our downstream systems by ensuring that package dependencies are resolved in the same way during installation.

2. Install this repo (`vcp-model-client-tools`) in your virtual environment:

    ```
    $ pip install .
    ```

3. Generate the directory structure in which you will implement the `mlflow` wrapper for your model. This will create a directory called `<model-name>_mlflow_pkg` where you will flesh out the implementation:

    ```
    $ python generate_model_pkg_dir_structure.py --model-name <model-name> --target <target-directory> --pkg-template mlflow_template --model-archetype file_uri_to_tensor
    ```

    Example:

    ```
    $ python generate_model_pkg_dir_structure.py --model-name transcriptformer --target ~/model_pkg_playground --pkg-template mlflow_template --model-archetype file_uri_to_tensor
    ```

    ```
    $ tree transcriptformer_mlflow_pkg

    transcriptformer_mlflow_pkg
    ├── mlflow_packager.py
    ├── generate_requirements_txt.py
    ├── model_code
    │   ├── __init__.py
    │   └── transcriptformer_mlflow_model.py
    ├── model_data
    ├── model_spec.py
    └── requirements.in
    ```

4. Install `uv` and `pip-tools` in the virtual environment: 

    ```
    $ pip install uv pip-tools
    ```
    **NOTES:**

    - We use `uv` for fast dependency resolution. `pip-tools` is slower but we use it as fallback.

5. Fill out the `requirements.in` file inside the generated directory.

    Example:

    ```
    $ cat requirements.in

    # Required dependencies - DO NOT EDIT THIS SECTION
    mlflow==3.1.0

    # Provide your dependencies below
    transcriptformer==0.3.0
    ```

    **NOTE:** Typically it is sufficient to just provide your pip installable
    model package or any top level packages in `requirements.in`. The next step will traverse the transitive dependencies of the packages listed in `requirements.in` and produce complete package dependency list in a `requirements.txt` file.

6. Generate a `requirements.txt` file from `requirements.in`. This will capture all the transitive dependencies of the packages you listed in `requirements.in`. **Provide the python version of your virtual environment as an argument.**

    Example:

    ```
    $ python generate_requirements_txt.py --requirements-in requirements.in --python-version 3.11
    ```

7. Install the packages frozen in `requirements.txt` in your current virtual environment:

    ```
    $ pip install -r requirements.txt
    ```

    **NOTES:**

    - We are installing packages in the `requirements.txt` because subsequent steps (ex: downloading model weights) may require tools bundled in your model's python package. 

    - We are using `pip` to install packages (instead of `uv`) because `uv` does not enforce the upper bound of versions in `requires-python` directive.
    
    - **Do NOT** use `pip install --no-deps -r requirements.txt` as this bypasses pip's dependency resolver. For example, if your model requires `flash-attn`, `--no-deps` flag will not check that `flash-attn` requires that `torch` be installed first and this might result in an error during installation!
    
8. Download all necessary artifacts (ex: model weights and auxiliary data) to the `model_data` directory.

    Example:

    ```
    $ transcriptformer download tf-sapiens --checkpoint-dir model_data/
    ```

# Specify the model's input/output schema

Specify the schema for the inputs and outputs of the model by filling out `model_spec.py`. You must specify two things in this file:

- A model signature that captures your input and output schema using [MLflow ModelSignature](https://mlflow.org/docs/latest/ml/model/signatures) data types.

- A real input example. The `input_uri` field should point to a real input file because the packaging script will verify correctness by running the input example through a forward pass.

The doc strings in `model_spec.py` give detailed guidance on how to fill in this file. See the **INSTRUCTIONS FOR CUSTOMIZATION** section in the module doc string.

# Create a **MLflow PythonModel** wrapper
1. Complete the implementation of the following methods in the `model_code/<model-name>_mlflow_model.py` file. The doc strings on these methods provide guidance on how to implement them:

    ```
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        raise NotImplementedError

    def _get_input(self, uri: str, **params) -> Any:
        raise NotImplementedError

    def _forward(self, input_obj: Any, **params) -> np.ndarray:
        raise NotImplementedError
    ```

# Create a **MLflow Model** package

1. Run `mlflow_packager.py` to create the **MLflow Model** artifact _for each model variant_. The default artifact path is `./mlflow_model_artifact` but you can use `--artifact-dir-path` to write the artifact to a different directory.


    See the thorough module docstring for `mlflow_packager.py` or get thorough CLI usage documentation by typing `python mlflow_packager.py --help`.

    **NOTE:**
    The script has a `--skip-inference` switch that will package the model and 
    skip the inference step. This switch is meant for local testing or for a CI script 
    that will run model packaging and inference in separate steps.

    We **HIGHLY RECOMMEND** you run the `mlflow_packager.py` script **without** the `--skip-inference` switch so that you can verify that the packaged model can indeed 
    run a forward pass!

    Example:

    ```
    $ python mlflow_packager.py --model-class model_code.transcriptformer_mlflow_model:TranscriptformerMLflowModel --artifact checkpoint=tf_sapiens --model-config-json '{"model_variant":"tf_sapiens"}' --model-tag model_variant=tf_sapiens
    ```

    Example Artifact Output Directory Structure:

    ```
    $ tree mlflow_model_artifact/

    mlflow_model_artifact/
    ├── MLmodel
    ├── artifacts
    │   └── tf_sapiens
    │       ├── config.json
    │       ├── model_weights.pt
    │       └── vocabs
    │           ├── assay_vocab.json
    │           └── homo_sapiens_gene.h5
    ├── conda.yaml
    ├── input_example.json
    ├── python_env.yaml
    ├── python_model.pkl
    ├── requirements.txt
    └── serving_input_example.json
    ```

# Use the **MLflow Model** to run inference locally!

1. Create a json input payload by copying `serving_input_example.json` from the `mlflow_model_artifact` into your current directory and modifying its contents appropriately:

    ```
    $ cp mlflow_model_artifact/serving_input_example.json .
    ```

    Example:

    ```
    $ cat serving_input_example.json

    {
      "dataframe_split": {
        "columns": [
          "input_uri"
        ],
        "data": [
          [
            "/home/user/datasets/example_small.h5ad"
          ]
        ]
      },
      "params": {
        "batch_size": 32,
        "precision": "16-mixed",
        "gene_col_name": "ensembl_id"
      }
    ```

2. Run inference:

    ```
    $ mlflow models predict --model-uri ./mlflow_model_artifact --content-type json --input-path serving_input_example.json --output-path test_output.json --env-manager <virtualenv-manager>
    ```

    **You must specify `conda` or `virtualenv` for `--env-manager`** to be compatible with our downstreams systems.

    Example Usage:

    ```
    $ mlflow models predict --model-uri ./mlflow_model_artifact --content-type json --input-path serving_input_example.json --output-path test_output.json --env-manager conda 
    ``` 

3. Verify the output by examining the contents of `test_output.json`.

    Example:

    ```
    $ head -c 1000 test_output.json

    {"predictions": [[-0.1429818570613861, -0.1260850429534912, 0.040420662611722946, -0.19157768785953522, 0.22647874057292938, 0.15660905838012695, -0.13574902713298798, 0.0721682459115982, 0.024216821417212486, -0.04561154171824455, -0.3007250130176544, 0.08299852907657623, 0.0005049569299444556, 0.003926896024495363, 0.08237997442483902, 0.18843598663806915, -0.008095022290945053, -0.012726053595542908, -0.11273445188999176, 0.03558430075645447, -0.050463370978832245, 0.18308386206626892, 0.07628823071718216, 0.017358627170324326, 0.027970347553491592, -0.33074551820755005, 0.0716106966137886, 0.15132226049900055, -0.0679834634065628, 0.06222778931260109, -0.13658343255519867, 0.2187119573354721, 0.1522756665945053, 0.023600872606039047, -0.1128731444478035, 0.08771258592605591, 0.045885197818279266, 0.05724436417222023, 0.16931083798408508, -0.1518353968858719, 0.03722844272851944, -0.13497602939605713, -0.015310755930840969, -0.10012456774711609, -0.03223331272602081, 0.2051643580198
    ```

