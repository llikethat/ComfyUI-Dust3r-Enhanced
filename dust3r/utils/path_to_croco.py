# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# CroCo submodule import
# Enhanced with better error handling for ComfyUI
# --------------------------------------------------------

import sys
import os
import os.path as path

def log(msg, level="INFO"):
    print(f"[Dust3r/CroCo] [{level}] {msg}")

HERE_PATH = path.normpath(path.dirname(__file__))
log(f"path_to_croco.py location: {HERE_PATH}")

# Try multiple possible locations for croco
POSSIBLE_CROCO_PATHS = [
    path.normpath(path.join(HERE_PATH, '../../croco')),  # Standard location
    path.normpath(path.join(HERE_PATH, '../../../croco')),  # Alternative
]

CROCO_REPO_PATH = None
CROCO_MODELS_PATH = None

for croco_path in POSSIBLE_CROCO_PATHS:
    models_path = path.join(croco_path, 'models')
    log(f"Checking for CroCo at: {croco_path}")
    log(f"  Models path: {models_path}")
    
    if path.isdir(models_path):
        CROCO_REPO_PATH = croco_path
        CROCO_MODELS_PATH = models_path
        log(f"  ✓ Found CroCo models!")
        break
    else:
        log(f"  ✗ Not found")

if CROCO_REPO_PATH and CROCO_MODELS_PATH:
    if CROCO_REPO_PATH not in sys.path:
        sys.path.insert(0, CROCO_REPO_PATH)
        log(f"Added to sys.path: {CROCO_REPO_PATH}")
    
    # Verify we can find the models
    log(f"CroCo models directory contents: {os.listdir(CROCO_MODELS_PATH)}")
else:
    error_msg = f"""
CroCo models not found!

Searched in:
{chr(10).join('  - ' + p for p in POSSIBLE_CROCO_PATHS)}

Please ensure the 'croco' folder with 'models' subfolder is present in the ComfyUI-Dust3r-Enhanced directory.

The folder structure should be:
ComfyUI-Dust3r-Enhanced/
├── croco/
│   └── models/
│       ├── croco.py
│       ├── blocks.py
│       └── ...
├── dust3r/
└── ...
"""
    log(error_msg, "ERROR")
    raise ImportError(error_msg)
