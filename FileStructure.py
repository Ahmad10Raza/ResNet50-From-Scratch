import os
from pathlib import Path

def create_project_structure(base_dir=""):
    """Create the directory structure for Hugging Face compatible deployment."""
    
    structure = {
        "data": {
            "imagenet": [],
            "processed": [],
            "samples": [],
        },
        "notebooks": [
            "01_resnet50_scratch.ipynb",
            "02_data_pipeline.ipynb",
            "03_training_analysis.ipynb",
        ],
        "src": [
            "config.py",
            "data_loader.py",
            "model.py",
            "train.py",
        ],
        "saved_models": {
            "weights": [],
            "full_model": [],
        },
        "streamlit_utils": {  # Changed from list to dict
            "utils.py": None,  # File marker
            "assets": [],      # Subdirectory
        },
        "root_files": [
            "app.py",  # Now in root for Hugging Face
            "requirements.txt",
            "README.md",
        ]
    }

    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    print(f"Created base directory: {base_path}")

    # Create app.py in root
    (base_path / "app.py").touch()
    print("Created app.py in root directory")

    # Create other directories
    for dir_name, contents in structure.items():
        if dir_name == "root_files":
            continue
            
        dir_path = base_path / dir_name
        if isinstance(contents, dict):
            dir_path.mkdir(exist_ok=True)
            for item_name, item_contents in contents.items():
                item_path = dir_path / item_name
                if isinstance(item_contents, list):  # It's a directory
                    item_path.mkdir(exist_ok=True)
                    print(f"Created directory: {item_path}")
                else:  # It's a file marker (None)
                    item_path.touch()
                    print(f"Created file: {item_path}")
        elif isinstance(contents, list):
            dir_path.mkdir(exist_ok=True)
            for item in contents:
                if isinstance(item, str):
                    (dir_path / item).touch()
                    print(f"Created file: {dir_path/item}")

    # Create remaining root files
    for file in structure["root_files"]:
        if file != "app.py":  # Already created
            (base_path / file).touch()
            print(f"Created root file: {file}")

    print("\nHugging Face compatible structure created!")
    print(f"Location: {base_path.absolute()}")

if __name__ == "__main__":
    create_project_structure()