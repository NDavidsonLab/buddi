"""
Tests for example notebooks using papermill.

Note that you must explicitly invoke pytest with --run-slow to execute these
tests, as it takes ~30 minutes to run all notebooks.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import List

import pytest

REPO_ROOT = Path(__file__).parent.parent


def get_example_notebooks() -> List[Path]:
    """Discover all .ipynb files in examples directory and subdirectories."""
    examples_dir = Path(__file__).parent.parent / "examples"

    # find all notebook files, excluding checkpoints
    notebooks = [
        nb
        for nb in examples_dir.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in nb.parts
    ]

    return sorted(notebooks)


def run_notebook_with_papermill(
    notebook_path: Path,
    kernel_name: str = "buddi",
) -> bool:
    """
    Run a notebook using papermill in a headless manner.

    Args:
        notebook_path: Path to the notebook to run
        kernel_name: Name of the Jupyter kernel to use

    Returns:
        True if successful, False otherwise

    Raises:
        RuntimeError: If papermill execution fails
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / notebook_path.name

        cmd = [
            "python",
            "-m",
            "papermill",
            str(notebook_path),
            str(output_path),
            "--kernel",
            kernel_name,
        ]

        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Notebook {notebook_path.name} failed to execute.\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        return True


class TestNotebooks:
    """Test suite for running example notebooks."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "notebook_path",
        get_example_notebooks(),
        ids=lambda x: str(x.relative_to(Path(__file__).parent.parent / "examples")),
    )
    def test_notebook_runs_without_errors(self, notebook_path: Path):
        """Test that each notebook runs to completion without errors."""
        run_notebook_with_papermill(notebook_path)
