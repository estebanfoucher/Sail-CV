from pathlib import Path
import json
import yaml
from typing import Dict

def load_parameters(specs_file: str) -> Dict:
    """Load checkerboard specifications from JSON or YAML file."""
    specs_path = Path(specs_file)
    if specs_path.suffix.lower() in ['.json']:
        with open(specs_path, 'r') as f:
            return json.load(f)
    elif specs_path.suffix.lower() in ['.yml', '.yaml']:
        with open(specs_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {specs_path.suffix}")