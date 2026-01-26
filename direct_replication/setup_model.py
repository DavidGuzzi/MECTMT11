"""
Setup script to copy essential Dynare model files from repo/ to model/
Run this once before executing the replication notebook.
"""
import shutil
from pathlib import Path

def setup_model_directory():
    """Copy essential model files from repo/ to direct_replication/model/"""

    # Define paths
    project_root = Path(__file__).parent.parent
    repo_dir = project_root / 'repo'
    model_dir = Path(__file__).parent / 'model'

    # Create model directory if it doesn't exist
    model_dir.mkdir(exist_ok=True)
    print(f"Created/verified directory: {model_dir}")

    # Essential files to copy (4 required files)
    essential_files = [
        'usmodel.mod',           # Main DSGE model
        'usmodel_stst.m',        # Steady-state solver
        'usmodel_data.mat',      # Data for estimation
        'usmodel_mode.mat',      # Prior parameter estimates
    ]

    # Copy files
    for filename in essential_files:
        src = repo_dir / filename
        dst = model_dir / filename

        if not src.exists():
            print(f"WARNING: Source file not found: {src}")
            continue

        # Copy file (overwrite if exists)
        shutil.copy2(src, dst)
        print(f"Copied: {filename}")

    print(f"\nSetup complete! Model files copied to: {model_dir}")
    print("\nFiles in model directory:")
    for f in sorted(model_dir.glob('*')):
        print(f"  - {f.name}")

    return model_dir

if __name__ == '__main__':
    setup_model_directory()
