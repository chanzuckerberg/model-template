# Model Card Template
<!-- You can use standard [Markdown](https://www.markdownguide.org/basic-syntax/) in this file to format your responses including lists, tables, links, and headings.

To include images in your model card, place them in the `model_card_images/` folder and reference them like so:

![Descriptive alt text that can also serve as a caption](./model_card_images/your_image.png)

Write descriptive alt text that explains what's in the image for screen readers. This is crucial for users with visual impairments. For guidance:
- [WebAIM Alt Text Guide](https://webaim.org/techniques/alttext/)
- [W3C Image Decision Tree](https://www.w3.org/WAI/tutorials/images/decision-tree/) -->

<!-- MODEL DETAILS SECTION -->

## Model Details

### Model Architecture
<!-- Brief description of the model architecture (e.g., number of layers and attention heads, embedding dimensions, input size or context length) and rationale behind it -->

...

### Parameters
<!-- Number of parameters (e.g., 15 million) -->

...

### Citation
<!-- Provide citation information for users of the model -->

...

<!-- INTENDED USE SECTION -->

## Intended Use
<!-- This section addresses questions around how the model is intended to be used in different applied contexts, discusses the foreseeable users of the model (including those affected by the model), and describes uses that are considered out of scope or misuse of the model. -->

### Primary Use Cases
<!-- List primary use cases (e.g., cell type classification, perturbation prediction, protein localization, cell morphology profiling). You can include the 'tasks_performed_by_model' that you provided in model_card_details.yaml as a starting point.  -->

...

### Out-of-Scope or Unauthorized Use Cases
<!-- Suggested Text:

"Do not use the model for the following purposes:
 - Use that violates applicable laws, regulations (including trade compliance laws), or third party rights such as
   privacy or intellectual property rights.
 - Any use that is prohibited by the [link to model license] license.
 - Any use that is prohibited by the Acceptable Use Policy."

[Please include other specific out-of-scope use cases that may be relevant for this model, as applicable]
-->

...

<!-- TRAINING DETAILS SECTION -->

## Training Details
<!-- This section provides information to describe and replicate training, including the training data and the speed and size of training elements. -->

...

### Training Procedure
<!-- Briefly describe the training approach including data pre-processing steps (e.g., steps taken to clean and preprocess the data, detail tokenization, modality dependent resizing/rewriting) -->

...

### Training Code
<!-- (optional, but strongly encouraged) Provide links to training scripts -->

...

### Speeds, Sizes, Times
<!-- (optional, include if available) Provide information about throughput, start/end time, checkpoint size if relevant, etc. (optional, include if available) -->

...

### Training Hyperparameters
<!-- (optional, include if available) Examples: fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision  -->

...

<!-- PERFORMANCE METRICS SECTION -->

## Performance Metrics
<!-- This section describes the evaluation protocols, what is being measured in the evaluation, and provides the results. -->

### Metrics
<!-- List evaluation metrics used and the rationale for using them along with links where applicable. If this model was benchmarked against existing models, list them here and explain the rationale for comparison.

For example:
"The model was evaluated using a range of benchmarks to measure its performance.
Key metrics include: [metrics]." -->

...

### Evaluation Datasets
<!-- List the evaluation datasets along with links to the data where possible. Link to evaluation datasets processing code if available. -->

...

### Evaluation Results
<!-- Provide table and/or figures summarizing evaluation results -->

...

<!-- BIASES, RISKS, AND LIMITATIONS SECTION -->

## Biases, Risks, and Limitations
<!-- This section identifies potential harms, misunderstandings, and technical and limitations. It also provides
information on warnings and potential mitigations. Suggestions are provided below. -->

### Potential Biases
<!-- Suggested Text:

"- The model may reflect biases present in the training data.
 - Certain demographic groups may be underrepresented."

[Please include other specific biases that may be relevant for this model, as applicable] -->

...

### Risks
<!-- Suggested Text:

"Areas of risk may include but are not limited to:
 - Inaccurate outputs or hallucinations
 - Potential misuse for incorrect biological interpretations."

[Please include other specific risks that may be relevant for this model, as applicable] -->

...

### Limitations
<!--
Suggested Text:

"- The model may not perform well on general tasks."

[Please include other specific limitations that may be relevant for this model, as applicable]
-->

...

### Caveats and Recommendations
<!-- (optional)
Suggested Text:

"- Review and validate outputs generated by the model.
 - We are committed to advancing the responsible development and use of artificial intelligence. Please follow our Acceptable Use Policy when using the model."

For CZI and CZ Biohub models:
"- Should you have any security or privacy issues or questions related to the model, please reach out to our team at security@chanzuckerberg.com or privacy@chanzuckerberg.com, respectively."

[Please include other recommendations that may be relevant for users of this model, as applicable]
-->

...

<!-- ACKNOWLEDGEMENTS -->

## Acknowledgements
<!-- (optional) This section is for providing acknowledgement of other contributors or supporting organizations. -->

...
