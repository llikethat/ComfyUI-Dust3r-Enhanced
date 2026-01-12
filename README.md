# ComfyUI-Dust3r-Enhanced

A ComfyUI custom node for **DUSt3R** (Dense and Unconstrained Stereo 3D Reconstruction) with enhanced error handling and verbose logging for easier troubleshooting.

## Features

- 🔧 **Enhanced Error Handling**: Detailed error messages and logging to help diagnose issues
- 📊 **Verbose Logging**: Step-by-step progress output during model loading and reconstruction
- 🎯 **Multiple Nodes**: 
  - `Dust3r Model Loader` - Load checkpoints separately
  - `Dust3r Reconstruct 3D` - Run reconstruction with full control
  - `Dust3r Simple` - All-in-one convenience node
- 🖼️ **Output Visualization**: Returns depth maps and confidence maps as images
- 📦 **GLB Export**: Outputs 3D scene as GLB file for viewing in 3D viewers

## Installation

### 1. Clone or Download

Place this entire folder in your ComfyUI custom_nodes directory:

```bash
cd /path/to/ComfyUI/custom_nodes/
git clone https://github.com/your-repo/ComfyUI-Dust3r-Enhanced.git
```

Or download and extract to:
```
ComfyUI/custom_nodes/ComfyUI-Dust3r-Enhanced/
```

### 2. Install Dependencies

```bash
cd ComfyUI-Dust3r-Enhanced
pip install -r requirements.txt
```

### 3. Download Checkpoints

Download Dust3r model checkpoints from the [official repository](https://github.com/naver/dust3r#checkpoints):

| Model | Size | Link |
|-------|------|------|
| DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth | ~1.1GB | [Download](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth) |
| DUSt3R_ViTLarge_BaseDecoder_512_linear.pth | ~1.1GB | [Download](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth) |
| DUSt3R_ViTLarge_BaseDecoder_224_linear.pth | ~1.1GB | [Download](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth) |

Place downloaded checkpoints in:
```
ComfyUI-Dust3r-Enhanced/checkpoints/
```

### 4. Restart ComfyUI

**Important**: You must restart ComfyUI after adding checkpoints for them to be detected.

## Directory Structure

```
ComfyUI-Dust3r-Enhanced/
├── __init__.py          # Main entry point
├── nodes.py             # ComfyUI node definitions
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── checkpoints/        # Place your .pth files here
│   └── DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
├── input/              # Temporary input images (auto-created)
├── output/             # Output GLB files (auto-created)
├── dust3r/             # Core Dust3r library
│   ├── model.py
│   ├── inference.py
│   ├── cloud_opt/
│   ├── heads/
│   ├── utils/
│   └── viz.py
└── croco/              # CroCo encoder (required)
    └── models/
        ├── croco.py
        ├── blocks.py
        └── ...
```

## Nodes

### Dust3r Model Loader

Loads a Dust3r checkpoint and returns the model.

**Inputs:**
- `checkpoint`: Select from available checkpoint files
- `device`: `cuda`, `cpu`, or `auto`

**Outputs:**
- `model`: DUST3R_MODEL type for use with reconstruction node

### Dust3r Reconstruct 3D

Runs 3D reconstruction on input images.

**Inputs:**
- `model`: DUST3R_MODEL from loader
- `images`: IMAGE batch (2+ images recommended)
- `image_size`: Target size (224-1024, default 512)
- `scene_graph`: `complete`, `swin`, or `oneref`
- `schedule`: `linear` or `cosine`
- `iterations`: Optimization iterations (0-5000)
- `min_conf_threshold`: Confidence threshold (0-20)
- `cam_size`: Camera visualization size
- `as_pointcloud`: Output as point cloud instead of mesh
- `mask_sky`: Remove sky from reconstruction
- `clean_depth`: Clean up depth artifacts
- `transparent_cams`: Make cameras transparent in output

**Outputs:**
- `glb_path`: Path to exported GLB file
- `depth_maps`: Depth visualization images
- `confidence_maps`: Confidence visualization images

### Dust3r Simple (All-in-One)

Convenience node that combines loading and reconstruction.

**Inputs:**
- `images`: IMAGE batch
- `checkpoint`: Select checkpoint
- `image_size`: Target size (optional)
- `iterations`: Optimization iterations (optional)

**Outputs:**
- `glb_path`: Path to exported GLB file

## Troubleshooting

### "NO_CHECKPOINTS_FOUND" Error

1. Ensure checkpoint files are in the `checkpoints/` folder
2. Restart ComfyUI (checkpoints are scanned at startup)
3. Check console output for checkpoint detection logs

### Import Errors

Check the console for detailed error messages. Common issues:
- Missing dependencies: Run `pip install -r requirements.txt`
- Missing CroCo: Ensure `croco/models/` folder exists with all files

### CUDA Out of Memory

- Reduce `image_size` (try 224 or 384)
- Reduce number of input images
- Use `cpu` device (slower but uses less VRAM)

### Model Loading Fails

- Verify checkpoint file is not corrupted (re-download if needed)
- Check file size matches expected (~1.1GB for ViTLarge models)
- Look at console for specific error messages

## Console Output

This node provides detailed console output. Look for lines starting with:
- `[Dust3r]` - General node messages
- `[Dust3r/Model]` - Model loading messages
- `[Dust3r/CroCo]` - CroCo import messages

Log levels:
- `[INFO]` - Normal operation
- `[WARNING]` - Potential issues
- `[ERROR]` - Failures

## Credits

- [DUSt3R](https://github.com/naver/dust3r) by Naver Labs
- [CroCo](https://github.com/naver/croco) by Naver Labs
- Original ComfyUI implementation inspiration

## License

This wrapper is provided as-is. DUSt3R is licensed under CC BY-NC-SA 4.0 (non-commercial use only).
