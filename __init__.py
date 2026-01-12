"""
ComfyUI-Dust3r-Enhanced
A ComfyUI custom node for DUSt3R (Dense and Unconstrained Stereo 3D Reconstruction)
With enhanced error handling and verbosity for easier troubleshooting.
"""

import os
import sys

# Get the directory containing this file
NODE_DIR = os.path.dirname(os.path.abspath(__file__))

def log(msg, level="INFO"):
    """Simple logging function with levels"""
    print(f"[Dust3r-Enhanced] [{level}] {msg}")

log(f"Initializing from: {NODE_DIR}")

# Add this directory to path for imports
if NODE_DIR not in sys.path:
    sys.path.insert(0, NODE_DIR)
    log(f"Added to sys.path: {NODE_DIR}")

# Add croco models to path
CROCO_PATH = os.path.join(NODE_DIR, "croco")
CROCO_MODELS_PATH = os.path.join(CROCO_PATH, "models")

if os.path.isdir(CROCO_MODELS_PATH):
    if CROCO_PATH not in sys.path:
        sys.path.insert(0, CROCO_PATH)
        log(f"Added CroCo to sys.path: {CROCO_PATH}")
else:
    log(f"WARNING: CroCo models not found at {CROCO_MODELS_PATH}", "WARNING")
    log("Please ensure the 'croco' folder with 'models' subfolder is present", "WARNING")

# Setup checkpoints directory
CHECKPOINTS_DIR = os.path.join(NODE_DIR, "checkpoints")
if not os.path.exists(CHECKPOINTS_DIR):
    os.makedirs(CHECKPOINTS_DIR)
    log(f"Created checkpoints directory: {CHECKPOINTS_DIR}")

# List available checkpoints
def get_available_checkpoints():
    """Get list of available checkpoint files"""
    checkpoints = []
    if os.path.exists(CHECKPOINTS_DIR):
        for f in os.listdir(CHECKPOINTS_DIR):
            if f.endswith(('.pth', '.pt', '.safetensors')):
                checkpoints.append(f)
    log(f"Found {len(checkpoints)} checkpoint(s): {checkpoints}")
    return checkpoints if checkpoints else ["NO_CHECKPOINTS_FOUND"]

# Import nodes
try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    log(f"Successfully loaded nodes: {list(NODE_CLASS_MAPPINGS.keys())}")
except Exception as e:
    log(f"Failed to import nodes: {e}", "ERROR")
    import traceback
    traceback.print_exc()
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = "./web"
