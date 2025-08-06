#!/usr/bin/env python3
"""
Generate pip-compatible requirements.txt using uv pip compile with automatic fallback.

This script attempts to use the speed of `uv pip compile` while ensuring compatibility
with `pip` as the installer. It automatically detects Python version compatibility
issues and generates constraints to resolve them. If `uv` fails due to dependency
resolution issues, it falls back to `pip-compile` for reliability.

Key Features:
- Fast dependency resolution with `uv pip compile`
- Automatic Python version compatibility checking
- Dynamic constraint generation for incompatible packages
- Fallback to `pip-compile` when `uv` fails
- Validation with `pip install --dry-run`
- Debug mode for inspection of intermediate files

Usage Examples:
    # Basic usage
    python generate_requirements_txt.py --requirements-in requirements.in --python-version 3.11

    # With debug mode to inspect intermediate files
    python generate_requirements_txt.py --requirements-in requirements.in --python-version 3.11 --debug

    # With additional constraints file
    python generate_requirements_txt.py --requirements-in requirements.in --python-version 3.11 --constraints additional_constraints.txt
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

# Import packaging for robust version parsing
from packaging import version as packaging_version
from packaging.specifiers import SpecifierSet


def setup_logging(verbose: bool = False) -> None:
    """
    Set up logging configuration.

    Args:
        verbose: Whether to enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def normalize_python_version(python_version: str) -> str:
    """
    Normalize Python version to major.minor format.

    Args:
        python_version: Python version string (e.g., "3.11.11", "3.11.11.alpha1")

    Returns:
        Normalized version string (e.g., "3.11")
    """
    try:
        # Parse the version and extract major.minor
        version_obj = packaging_version.parse(python_version)
        return f"{version_obj.major}.{version_obj.minor}"
    except Exception as e:
        logging.warning(f"Could not parse Python version '{python_version}': {e}")
        # Fallback: try to extract major.minor manually
        parts = python_version.split(".")
        if len(parts) >= 2:
            return f"{parts[0]}.{parts[1]}"
        return python_version


def get_python_version() -> str:
    """Get the current Python version as a string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def run_uv_compile(requirements_in: str, output_file: str) -> bool:
    """
    Run uv pip compile to generate initial requirements.txt.

    Args:
        requirements_in: Path to requirements.in file
        output_file: Path to output requirements.txt file

    Returns:
        True if successful, False otherwise
    """
    logging.info(f"Running uv pip compile: {requirements_in} -> {output_file}")
    try:
        cmd = ["uv", "pip", "compile", requirements_in, "-o", output_file]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(f"uv pip compile failed: {result.stderr}")
            return False

        logging.info("uv pip compile completed successfully")
        return True

    except Exception as e:
        logging.error(f"Exception during uv pip compile: {e}")
        return False


def parse_requirements(requirements_file: str) -> dict:
    """
    Parse requirements.txt file to extract package names and versions.

    Args:
        requirements_file: Path to requirements.txt file

    Returns:
        Dictionary mapping package names to versions
    """
    packages = {}

    try:
        with open(requirements_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "==" in line:
                    # Extract package name and version
                    parts = line.split("==")
                    if len(parts) == 2:
                        package_name = parts[0].strip()
                        version = parts[1].strip()
                        packages[package_name] = version
        logging.debug(f"Parsed {len(packages)} packages from {requirements_file}")
    except Exception as e:
        logging.warning(f"Could not parse requirements file {requirements_file}: {e}")

    return packages


def check_python_compatibility(
    package_name: str, version: str, python_version: str
) -> bool:
    """
    Check if a package version is compatible with the given Python version.

    This function queries PyPI for package metadata and checks the `requires_python`
    field to determine compatibility.

    Args:
        package_name: Name of the package
        version: Version of the package
        python_version: Target Python version (e.g., "3.11")

    Returns:
        True if compatible, False otherwise
    """
    try:
        # Query PyPI for package metadata to get requires_python
        cmd = ["curl", "-s", f"https://pypi.org/pypi/{package_name}/{version}/json"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return True  # Assume compatible if we can't check

        data = json.loads(result.stdout)

        # Check requires_python field
        requires_python = data.get("info", {}).get("requires_python")
        if not requires_python:
            return True  # No Python requirement specified, assume compatible

        # Use packaging to properly parse and check version compatibility
        try:
            specifier_set = SpecifierSet(requires_python)
            python_version_obj = packaging_version.parse(python_version)
            return python_version_obj in specifier_set
        except Exception as e:
            logging.warning(
                f"Could not parse version specifier '{requires_python}' for {package_name}=={version}: {e}"
            )
            return True  # Assume compatible if we can't parse

    except Exception as e:
        logging.warning(
            f"Could not check compatibility for {package_name}=={version}: {e}"
        )
        return True  # Assume compatible if we can't check


def find_compatible_version(package_name: str, python_version: str) -> Optional[str]:
    """
    Find a compatible version of a package for the given Python version.

    This function queries PyPI for all available versions and finds the first
    one that is compatible with the target Python version.

    Args:
        package_name: Name of the package
        python_version: Target Python version (e.g., "3.11")

    Returns:
        Compatible version string if found, None otherwise
    """
    try:
        # Get all versions from PyPI
        cmd = ["curl", "-s", f"https://pypi.org/pypi/{package_name}/json"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        releases = data.get("releases", {})

        # Sort versions (newest first) using packaging
        versions = sorted(
            releases.keys(), key=lambda v: packaging_version.parse(v), reverse=True
        )

        # Find the first compatible version
        for version in versions:
            if check_python_compatibility(package_name, version, python_version):
                return version

        return None

    except Exception as e:
        logging.warning(f"Could not find compatible version for {package_name}: {e}")
        return None


def generate_constraints(requirements_file: str, python_version: str) -> List[str]:
    """
    Generate constraints for Python compatibility.

    This function analyzes all packages in the requirements.txt file and
    generates constraints for packages that are incompatible with the
    target Python version.

    Args:
        requirements_file: Path to requirements.txt file
        python_version: Target Python version (e.g., "3.11")

    Returns:
        List of constraint strings (e.g., ["package==version"])
    """
    packages = parse_requirements(requirements_file)
    constraints = []

    logging.info(
        f"Analyzing {len(packages)} packages for Python {python_version} compatibility..."
    )

    for package_name, version in packages.items():
        logging.debug(f"Checking {package_name}=={version}...")

        if not check_python_compatibility(package_name, version, python_version):
            logging.warning(
                f"{package_name}=={version} is NOT compatible with Python {python_version}"
            )

            # Find a compatible version
            compatible_version = find_compatible_version(package_name, python_version)
            if compatible_version:
                logging.info(
                    f"Found compatible version: {package_name}=={compatible_version}"
                )
                constraints.append(f"{package_name}=={compatible_version}")
            else:
                logging.error(f"Could not find compatible version for {package_name}")
        else:
            logging.debug(
                f"{package_name}=={version} is compatible with Python {python_version}"
            )

    return constraints


def run_uv_with_constraints(
    requirements_in: str, constraints_file: str, output_file: str
) -> bool:
    """
    Run uv pip compile with constraints.

    Args:
        requirements_in: Path to requirements.in file
        constraints_file: Path to constraints.txt file
        output_file: Path to output requirements.txt file

    Returns:
        True if successful, False otherwise
    """
    logging.info(
        f"Running uv pip compile with constraints: {requirements_in} + {constraints_file} -> {output_file}"
    )
    try:
        cmd = [
            "uv",
            "pip",
            "compile",
            requirements_in,
            "-o",
            output_file,
            "-c",
            constraints_file,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(f"uv pip compile with constraints failed: {result.stderr}")
            return False

        logging.info("uv pip compile with constraints completed successfully")
        return True

    except Exception as e:
        logging.error(f"Exception during uv pip compile with constraints: {e}")
        return False


def run_pip_compile(requirements_in: str, output_file: str) -> bool:
    """
    Run pip-compile as a fallback when uv fails.

    Args:
        requirements_in: Path to requirements.in file
        output_file: Path to output requirements.txt file

    Returns:
        True if successful, False otherwise
    """
    logging.info(f"Running pip-compile as fallback: {requirements_in} -> {output_file}")
    try:
        cmd = ["pip-compile", requirements_in, "-o", output_file]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(f"pip-compile failed: {result.stderr}")
            return False

        logging.info("pip-compile completed successfully")
        return True

    except Exception as e:
        logging.error(f"Exception during pip-compile: {e}")
        return False


def test_pip_install(requirements_file: str) -> bool:
    """
    Test if requirements.txt can be installed with pip.

    Args:
        requirements_file: Path to requirements.txt file

    Returns:
        True if pip dry-run succeeds, False otherwise
    """
    logging.info(f"Running pip dry-run test: {requirements_file}")
    try:
        cmd = ["pip", "install", "-r", requirements_file, "--dry-run"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logging.info("Pip dry-run successful - requirements.txt is compatible!")
            return True
        else:
            logging.error(f"Pip dry-run failed: {result.stderr}")
            return False

    except Exception as e:
        logging.error(f"Exception during pip dry-run: {e}")
        return False


def copy_debug_files(temp_dir: str, debug: bool) -> None:
    """
    Copy intermediate files to current directory for debugging.

    Args:
        temp_dir: Path to temporary directory containing debug files
        debug: Whether debug mode is enabled
    """
    if not debug:
        return

    try:
        debug_files = [
            ("requirements_initial.txt", "requirements_debug_uv_first_pass.txt"),
            ("requirements_final.txt", "requirements_debug_uv_final.txt"),
            ("constraints.txt", "constraints_debug_uv.txt"),
            ("requirements_pip_final.txt", "requirements_debug_pip_final.txt"),
        ]

        for src_name, dest_name in debug_files:
            src_path = os.path.join(temp_dir, src_name)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dest_name)
                logging.info(f"Debug: Copied {src_name} to {dest_name}")

    except Exception as e:
        logging.warning(f"Could not copy debug files: {e}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate pip-compatible requirements.txt using uv with automatic fallback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --requirements-in requirements.in --python-version 3.11
  %(prog)s --requirements-in requirements.in --python-version 3.11 --debug
  %(prog)s --requirements-in requirements.in --python-version 3.11 --constraints additional_constraints.txt
        """,
    )

    # Required arguments
    parser.add_argument(
        "--requirements-in", required=True, help="Path to requirements.in file"
    )
    parser.add_argument(
        "--python-version", required=True, help="Target Python version (e.g., 3.11)"
    )

    # Optional arguments
    parser.add_argument("--constraints", help="Additional constraints file to use")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Copy intermediate files to current directory for debugging",
    )
    parser.add_argument(
        "--keep-temp", action="store_true", help="Keep temporary files (for debugging)"
    )

    return parser.parse_args()


def main() -> int:
    """
    Main function that orchestrates the requirements.txt generation process.

    The process follows these steps:
    1. Generate initial requirements.txt with uv
    2. Test if it works with pip dry-run
    3. If not, analyze for Python compatibility issues
    4. Generate constraints and try uv again
    5. If uv fails, fall back to pip-compile
    6. Validate final result with pip dry-run
    7. Copy files to current directory if validation passes

    Returns:
        0 on success, 1 on failure
    """
    args = parse_arguments()

    # Set up logging
    setup_logging(args.verbose)

    # Normalize Python version to major.minor format for compatibility checking
    normalized_python_version = normalize_python_version(args.python_version)
    if normalized_python_version != args.python_version:
        logging.info(
            f"Normalized Python version from {args.python_version} to {normalized_python_version} for compatibility checking"
        )

    # Determine output filenames (will be set based on which tool succeeds)
    output_file = "requirements.txt"  # Default for non-debug mode
    constraints_output_file = "constraints.txt"  # Default for non-debug mode

    logging.info("🚀 Starting uv-to-pip requirements.txt generation")
    logging.info(
        f"Python version: {args.python_version} (using {normalized_python_version} for compatibility)"
    )
    logging.info(f"Input file: {args.requirements_in}")
    logging.info(f"Output file: {output_file}")

    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        logging.debug(f"Using temporary directory: {temp_dir}")

        # Step 1: Generate initial requirements.txt with uv
        logging.info("Step 1: Generating initial requirements.txt with uv...")
        initial_requirements = temp_dir_path / "requirements_initial.txt"

        if not run_uv_compile(args.requirements_in, str(initial_requirements)):
            logging.error("Failed to generate initial requirements.txt with uv")
            return 1

        # Step 2: Test if initial requirements.txt works with pip
        logging.info("Step 2: Testing initial requirements.txt with pip dry-run...")
        if test_pip_install(str(initial_requirements)):
            # Success! Copy initial requirements to output
            shutil.copy2(initial_requirements, output_file)
            logging.info(f"Success! Copied initial requirements.txt to {output_file}")
            copy_debug_files(temp_dir, args.debug)
            return 0

        # Step 3: Analyze Python compatibility and generate constraints
        logging.info("Step 3: Analyzing Python compatibility...")
        constraints = generate_constraints(
            str(initial_requirements), normalized_python_version
        )

        if not constraints:
            logging.error("No constraints generated, but pip dry-run failed")
            return 1

        # Add any additional constraints from file
        if args.constraints and os.path.exists(args.constraints):
            logging.info(f"Loading additional constraints from {args.constraints}")
            with open(args.constraints, "r") as f:
                additional_constraints = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
                constraints.extend(additional_constraints)
                logging.info(
                    f"Added {len(additional_constraints)} additional constraints"
                )

        # Write constraints file
        constraints_file = temp_dir_path / "constraints.txt"
        with open(constraints_file, "w") as f:
            f.write(
                f"# Auto-generated constraints for Python {normalized_python_version}\n"
            )
            for constraint in constraints:
                f.write(f"{constraint}\n")

        logging.info(f"Generated constraints for {len(constraints)} packages:")
        for constraint in constraints:
            logging.info(f"  {constraint}")

        # Step 4: Generate final requirements.txt with constraints
        logging.info("Step 4: Generating final requirements.txt with constraints...")
        final_requirements = temp_dir_path / "requirements_final.txt"

        if run_uv_with_constraints(
            args.requirements_in, str(constraints_file), str(final_requirements)
        ):
            # uv succeeded with constraints
            logging.info("uv succeeded with constraints")
            final_file = final_requirements
            used_uv = True
        else:
            # uv failed, try pip-compile
            logging.info("Step 4b: uv failed, trying pip-compile...")
            pip_final_requirements = temp_dir_path / "requirements_pip_final.txt"

            if run_pip_compile(args.requirements_in, str(pip_final_requirements)):
                logging.info("pip-compile succeeded")
                final_file = pip_final_requirements
                used_uv = False
            else:
                logging.error("Both uv and pip-compile failed")
                copy_debug_files(temp_dir, args.debug)
                return 1

        # Step 5: Validate final requirements.txt
        logging.info("Step 5: Validating final requirements.txt...")
        if test_pip_install(str(final_file)):
            # Determine correct output filename based on which tool was used
            if args.debug:
                if used_uv:
                    output_file = "requirements_debug_uv_final.txt"
                    constraints_output_file = "constraints_debug_uv.txt"
                else:
                    output_file = "requirements_debug_pip_final.txt"
                    # No constraints file for pip-compile
            else:
                output_file = "requirements.txt"
                constraints_output_file = "constraints.txt"

            # Success! Copy final requirements to output
            shutil.copy2(final_file, output_file)

            # Copy constraints file if uv was used
            if used_uv:
                shutil.copy2(constraints_file, constraints_output_file)
                logging.info(f"Success! Copied final requirements.txt to {output_file}")
                logging.info(f"Copied constraints to {constraints_output_file}")
            else:
                logging.info(
                    f"Success! Copied pip-compile requirements.txt to {output_file}"
                )
                logging.info("No constraints file (used pip-compile fallback)")

            copy_debug_files(temp_dir, args.debug)
            return 0
        else:
            logging.error("Final requirements.txt failed pip dry-run")
            copy_debug_files(temp_dir, args.debug)
            return 1


if __name__ == "__main__":
    sys.exit(main())
