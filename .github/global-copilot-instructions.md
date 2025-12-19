# AI Coding Assistant Global Instructions - Bioimaging Research Profile

## Developer Context
You're assisting a computational neuroscientist with 3+ years industry experience as data scientist/full-stack developer, now working as a bio-image analyst at a research institute. Focus on **fast development** and **practical solutions** over perfect code - time is always constrained with multiple concurrent projects.

## Technical Stack & Environment
- **Language**: Python (high-level, modern patterns)
- **Environment**: VSCode on Windows 10 with WSL2, transitioning to Linux
- **Package Management**: Pixi containers (not conda/pip)
- **Code Quality**: Ruff for linting/formatting (replaces black + isort)
- **Collaboration**: 2-person team using git with feature branches
- **Deployment**: HPC cluster with SLURM, Nextflow for job orchestration

## Code Style & Standards
- **PEP 8 compliant** with ruff enforcement
- **Readability over micro-optimizations** (but parallelize aggressively where beneficial)
- **Atomic, self-explanatory functions** (minimal docstrings except for wrapper functions)
- **Modern Python patterns**: pathlib, f-strings, type hints, dataclasses/Pydantic
- **Fail fast principle**: Clear error messages, explicit exception handling

## Documentation Framework
- **Documentation-as-Code**: Use MkDocs for all projects
- **Structure**: Follow the Diátaxis framework for organization
  - **Tutorials**: Learning-oriented (newcomers)
  - **How-to Guides**: Task-oriented (common operations)
  - **Explanations**: Understanding-oriented (concepts, architecture)
  - **References**: Information-oriented (technical details)

### Example Structure
```python
docs/
  index.md              # Overview and navigation
  tutorial/             # Step-by-step lessons
  how_to/               # Task-based procedures
  explanation/          # Concepts and architecture
  references/           # API details, parameters

### Error Handling Pattern
```python
from pathlib import Path

def safe_read_image(path: Path) -> np.ndarray:
    try:
        return imread(path)
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"Cannot read {path}: {e}")
        raise  # Let caller decide how to handle
    except Exception as e:
        logger.error(f"Unexpected error reading {path}: {e}")
        raise
```

## Architecture Preferences
- **Package-Based Architecture**: Clean project structure and architecture
- **Micro-services approach**: Reusable pipeline components
- **Functional pipelines** for simple cases, **dependency injection** for complex workflows
- **Main script pattern**: Single entry point calling modular components
- Always suggest **modern alternatives** and **industry standards**

### Pipeline Pattern Example
```python
from functools import partial

def pipeline(data, steps: list[Callable]):
    for step in steps:
        data = step(data)
    return data

# Usage
steps = [partial(resize_image, size=(512, 512)), segment_image, measure_objects]
result = pipeline(raw_image, steps)
```
## Hardware-Specific Optimizations
- **GPU Memory Management**: Use memory profiling and chunking
- **CPU/GPU Hybrid Processing**: Pre/post-processing on CPU, core algorithms on GPU
- **Streaming Processing**: Process data in streams instead of loading entirely

## Data & Performance Patterns
- **Data**: OME-Zarr format (TB-scale), microscopy images (TIF, STK, HDF5)
- **Storage**: HPC cluster filesystem, SQLite for metadata when beneficial
- **Memory**: Dask arrays for out-of-core processing
- **Parallelization**: joblib for per-node, SLURM array jobs for cluster-wide
- **I/O**: NGIO for file reading, pathlib for path handling

### HPC Parallelization
```python
from joblib import Parallel, delayed

def process_batch_on_node(image_paths: list[Path]):
    n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
    return Parallel(n_jobs=n_cores, backend='threading')(
        delayed(process_single_image)(path) for path in image_paths
    )
```

## Standardize "Human-in-the-Loop" Pattern
For projects requiring manual intervention (e.g., ROI selection, annotation, QC), use a dedicated `notebooks/` or `interactive/` directory. These notebooks should be treated as formal pipeline steps with clearly defined inputs and outputs.

- **Input:** Notebooks should read data from a standardized location (e.g., `processed_data/`).
- **Output:** Notebooks must write their results (e.g., CSVs of coordinates, annotation files) to a predictable location (e.g., `results/annotations/`) that downstream automated scripts can consume.
- **Parameters:** Use a tool like `papermill` to parameterize notebooks for easier integration into automated workflows if needed.

## Promote "Interactive Config Builder" Pattern
Every project should include a central, interactive configuration script (e.g., `build_config.py`). This script should guide the user, validate inputs, and generate static YAML configuration files for the pipeline. This avoids manual YAML editing errors and ensures all parameters for a run are captured.

- **Validation:** Use `Pydantic` to define strongly-typed configuration models. This provides immediate feedback on invalid values.
- **CLI:** Use `Typer` to create a user-friendly command-line interface for the configuration script.
- **Reproducibility:** The script's primary output should be a static `.yaml` file that is saved with the run's results, ensuring perfect reproducibility.

```python
# Example using Pydantic and Typer
import typer
from pydantic import BaseModel, Field, FilePath

class SegmentationConfig(BaseModel):
    model_path: FilePath
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)

def main(output_path: Path = typer.Option(...)):
    """Interactively builds and validates the pipeline config."""
    model_path = typer.prompt("Enter path to segmentation model")
    threshold = typer.prompt("Enter confidence threshold", type=float, default=0.5)
    
    config = SegmentationConfig(model_path=model_path, confidence_threshold=threshold)
    
    # Save the validated, static config
    with open(output_path, "w") as f:
        f.write(config.model_dump_json(indent=2))
    
    print(f"✅ Config saved to {output_path}")
```

## Refine Testing Strategy for Scientific Code
A pragmatic testing strategy for bio-image analysis should focus on reproducibility and correctness without excessive boilerplate.

- **Validation Data:** Each project repository must include a small, version-controlled dataset (~100MB) that can run through the entire pipeline in under 5 minutes. This is used for CI and local testing.
- **Integration Tests:** Create test scripts that assert the output of each pipeline step on the validation data matches an expected result.
    - *Example*: `assert num_segmented_objects == 15`
    - *Example*: `assert mean_intensity_after_norm > 0.4`
- **Visual "Golden" Tests:** For complex visual outputs (e.g., segmentation masks, plots), save a "golden" version of the output image/plot from the validation run. The test fails if the new output differs from the golden version, prompting a manual review. This catches unintended visual changes in algorithms.

## Key Libraries & Tools
- **Core**: NumPy, pandas, scikit-learn, OpenCV, scikit-image
- **Visualization**: matplotlib, napari
- **ML/Segmentation**: Segment Anything, StarDist, CellPose
- **File I/O**: NGIO (preferred), zarr, tifffile
- **Distributed**: Dask, Nextflow
- **CLI**: Typer (preferred over argparse)
- **Validation**: Pydantic for configs (fast to implement, prevents debugging time)

## Project Structure Patterns

### Hybrid Approach
Combine pipeline clarity with component reusability using a hybrid structure:
```
project/
├── pixi.toml              # Environment & task definitions
├── source/
│   ├── pipelines/         # Step-based pipeline definitions
│   │   ├── p01_cellseg/
│   │   └── p02_tissuequant/
│   ├── components/        # Reusable domain-specific modules
│   │   ├── io/            # Input/output operations
│   │   ├── preprocessing/ # Image preprocessing
│   │   ├── segmentation/  # Segmentation algorithms 
│   │   └── measurement/   # Feature extraction
│   └── utils/             # Cross-cutting utilities
├── tests/                 # Test directory mirroring source structure
│   ├── pipelines/
│   └── components/
├── cli/                   # Command-line interfaces
├── config/                # Configuration templates and schemas
├── notebooks/             # Exploratory and interactive analysis
├── runs/example/          # Working directory pattern
├── docs/                  # MkDocs documentation
└── .pre-commit-config.yaml # Ruff + basic checks
```

This structure provides several benefits:
- **Better code reuse**: Components are domain-focused and reusable across projects
- **Workflow clarity**: Pipelines assemble components into clear processing steps
- **Enhanced maintainability**: Specialized functionality is organized by domain
- **Nextflow compatibility**: Components map naturally to Nextflow processes
- **SLURM optimization**: Each component can specify resource requirements

### Legacy Step-Based Structure
For reference, the traditional step-based organization:
```
source/
├── s01_convert/       # Step-based organization
├── s02_segment/
└── s03_measure/
```

## Development Workflow
1. **Explore in Jupyter notebooks** → **Convert to scripts/modules**
2. **Feature branches** → **Merge after completion**
3. **Pixi environments** for reproducibility
4. **Pre-commit hooks** for code quality (suggest when missing)
5. **Minimal but effective testing** (focus on integration over unit tests)

### Quick Config Validation (High ROI)
```python
from pydantic import BaseModel, Field
from pathlib import Path

class Config(BaseModel):
    input_dir: Path
    n_jobs: int = Field(default=-1, ge=1)
    
    class Config:
        extra = "forbid"  # Catch config typos
```

## Communication Preferences
- **Concise first**, detailed if requested
- **Suggest modern alternatives** when applicable
- **Show clear examples** from bioimaging domain
- **Propose database integration** when metadata management would help
- **Focus on practical, implementable solutions**

## What NOT to suggest
- Complex OOP hierarchies for simple pipelines
- Extensive unit testing for exploratory analysis
- Over-engineered solutions for one-off scripts
- Generic advice - always relate to bioimaging/scientific computing context

## Key Patterns to Reinforce
- Use `pathlib.Path` not string paths
- Leverage `dask` for large image processing
- Implement logging with debug options
- Parallelize at multiple levels (per-file, per-job, per-cluster)
- Validate configs early to prevent late-stage failures
- Organize code for reusability across bioimaging projects

```nextflow
// Example Nextflow process using a component
process SEGMENT_CELLS {
    input:
        path image_file
    output:
        path "${image_file.baseName}_masks.zarr"
    
    script:
    """
    python -m components.segmentation.stardist_3d \
        --input ${image_file} \
        --output ${image_file.baseName}_masks.zarr
    """
}
```

```nextflow
profiles {
    standard {
        process {
            withName: SEGMENT_CELLS {
                cpus = 4
                memory = '16 GB'
            }
            withName: MEASURE_FEATURES { 
                cpus = 2
                memory = '8 GB'
            }
        }
    }
}
```

---

# Project-Specific Instructions

Extend global instructions with project-specific context: