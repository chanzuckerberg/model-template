#!/usr/bin/env python3
"""
Utility Script to render Jinja templates with or without sample data
Used to generate model card documentation files when making updates to templates
"""

from jinja2 import Environment, FileSystemLoader
import os

# Blank/default data matching the copier.yml structure
sample_data = {
    # Basic fields
    "model_display_name": "",
    "model_version": "v0.0.1",
    "primary_contact_email": "",
    "repository_link": "",
    "publication_preprint_link": "",
    "release_date": "",
    "model_modality": [],
    "model_description": "",
    "short_description": "",

    # Authors
    "authors": [],

    # Licenses
    "licenses": [],
    "license_name": "MIT",

    # Compute and architecture
    "compute_requirement": "",
    "system_requirements": "",
    "model_architecture_type": [],
    "tasks_performed_by_model": [],

    # Training details
    "training_date": "",
    "uses_synthetic_data": "",
    "uses_purchased_licensed_data": "",
    "flops_used": "",

    # Data types
    "input_data_type": [],
    "output_data_type": [],

    # Dataset sources
    "dataset_sources": [],

    # Links
    "quickstart_link": "",
    "tutorial_link": "",
    "model_download_link": "",

    # Detailed fields for model card details (using default "..." values)
    "parameters_count": "...",
    "model_citation": "...",
    "primary_use_cases": "...",
    "architecture_details": "...",
    "training_data": "...",
    "training_procedure": "...",
    "training_code": "...",
    "training_duration": "...",
    "training_hyperparameters": "...",
    "evaluation_metrics": "...",
    "evaluation_results": "...",
    "evaluation_datasets": "...",
    "out_of_scope_uses": "...",
    "potential_biases": "...",
    "risks": "...",
    "limitations": "...",
    "recommendations": "...",
    "acknowledgements": "..."
}

def render_template(template_path, output_path, data):
    """Render a Jinja template with the provided data"""
    # Get the directory and filename
    template_dir = os.path.dirname(template_path)
    template_file = os.path.basename(template_path)

    # Create Jinja environment
    env = Environment(loader=FileSystemLoader(template_dir))

    # Load and render template
    template = env.get_template(template_file)
    rendered = template.render(data)

    # Write output
    with open(output_path, 'w') as f:
        f.write(rendered)

    print(f"✓ Rendered {template_path}")
    print(f"  → Output: {output_path}\n")

    return rendered

if __name__ == "__main__":
    # Create output directory
    os.makedirs("rendered_output", exist_ok=True)

    print("Rendering Jinja templates with sample data...\n")

    # Render model card metadata
    metadata_output = render_template(
        "model_card_docs/model_card_metadata.yaml.jinja",
        "rendered_output/model_card_metadata.yaml",
        sample_data
    )

    # Render model card details
    details_output = render_template(
        "model_card_docs/model_card_details.md.jinja",
        "rendered_output/model_card_details.md",
        sample_data
    )

    print("=" * 60)
    print("All templates rendered successfully!")
    print("=" * 60)
    print(f"\nOutputs saved in: rendered_output/")
    print("- model_card_metadata.yaml")
    print("- model_card_details.md")
