"""
ComfyUI Dust3r Nodes with Enhanced Error Handling, Multi-Format Export, and Memory Optimization
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import traceback
import subprocess
import shutil
import gc
from typing import List, Tuple, Optional, Dict, Any

# ============================================================================
# LOGGING UTILITIES
# ============================================================================

class Logger:
    """Simple logger with color support"""
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'RESET': '\033[0m'      # Reset
    }
    
    @staticmethod
    def log(msg: str, level: str = "INFO"):
        color = Logger.COLORS.get(level, Logger.COLORS['INFO'])
        reset = Logger.COLORS['RESET']
        print(f"{color}[Dust3r] [{level}]{reset} {msg}")
    
    @staticmethod
    def debug(msg: str): Logger.log(msg, "DEBUG")
    
    @staticmethod
    def info(msg: str): Logger.log(msg, "INFO")
    
    @staticmethod
    def warning(msg: str): Logger.log(msg, "WARNING")
    
    @staticmethod
    def error(msg: str): Logger.log(msg, "ERROR")

log = Logger()

# ============================================================================
# MEMORY MANAGEMENT UTILITIES
# ============================================================================

class MemoryManager:
    """Manage GPU and CPU memory for processing long sequences"""
    
    @staticmethod
    def get_gpu_memory_info() -> Dict[str, float]:
        """Get GPU memory info in GB"""
        if not torch.cuda.is_available():
            return {"total": 0, "used": 0, "free": 0, "reserved": 0, "allocated": 0}
        
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        free = total - reserved
        
        return {
            "total": round(total, 2),
            "reserved": round(reserved, 2),
            "allocated": round(allocated, 2),
            "free": round(free, 2)
        }
    
    @staticmethod
    def get_cpu_memory_info() -> Dict[str, float]:
        """Get CPU memory info in GB"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "total": round(mem.total / (1024**3), 2),
                "available": round(mem.available / (1024**3), 2),
                "used": round(mem.used / (1024**3), 2),
                "percent": mem.percent
            }
        except ImportError:
            return {"total": 0, "available": 0, "used": 0, "percent": 0}
    
    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            log.debug("GPU memory cache cleared")
    
    @staticmethod
    def clear_all_memory():
        """Clear both GPU and CPU memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        log.debug("All memory caches cleared")
    
    @staticmethod
    def estimate_batch_size(num_images: int, image_size: int, gpu_memory_gb: float) -> int:
        """Estimate optimal batch size based on available GPU memory."""
        # Rough estimate: each image pair at 512x512 uses ~2GB VRAM
        mem_per_pair = 2.0 * (image_size / 512) ** 2
        
        # Leave 2GB headroom
        available = max(0.5, gpu_memory_gb - 2.0)
        
        max_pairs = int(available / mem_per_pair)
        batch_size = max(1, min(max_pairs, 4))
        
        return batch_size
    
    @staticmethod
    def log_memory_status():
        """Log current memory status"""
        gpu_info = MemoryManager.get_gpu_memory_info()
        cpu_info = MemoryManager.get_cpu_memory_info()
        
        log.info(f"GPU Memory: {gpu_info['allocated']:.1f}GB / {gpu_info['total']:.1f}GB "
                f"(Free: {gpu_info['free']:.1f}GB)")
        log.info(f"CPU Memory: {cpu_info['used']:.1f}GB / {cpu_info['total']:.1f}GB "
                f"({cpu_info['percent']}% used)")

mem_manager = MemoryManager()

# ============================================================================
# PATH SETUP
# ============================================================================

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(NODE_DIR, "checkpoints")
INPUT_DIR = os.path.join(NODE_DIR, "input")
OUTPUT_DIR = os.path.join(NODE_DIR, "output")

# Blender path - check multiple possible locations
BLENDER_PATHS = [
    "/workspace/ComfyUI/custom_nodes/ComfyUI-SAM3DBody/lib/blender/blender-4.2.3-linux-x64/blender",
    "/usr/bin/blender",
    "/usr/local/bin/blender",
    shutil.which("blender") or "",
]

BLENDER_PATH = None
for path in BLENDER_PATHS:
    if path and os.path.isfile(path):
        BLENDER_PATH = path
        log.info(f"Found Blender at: {BLENDER_PATH}")
        break

if not BLENDER_PATH:
    log.warning("Blender not found - format conversion will be limited to trimesh-supported formats")

# Supported output formats: (description, requires_blender)
SUPPORTED_FORMATS = {
    "glb": ("GLB (Binary glTF)", False),
    "gltf": ("glTF", False),
    "obj": ("Wavefront OBJ", False),
    "ply": ("PLY (Stanford)", False),
    "stl": ("STL", False),
    "off": ("OFF", False),
    "dae": ("Collada DAE", True),
    "fbx": ("FBX (Autodesk)", True),
    "blend": ("Blender", True),
    "usd": ("Universal Scene Description", True),
    "usdc": ("USD (Crate/Binary)", True),
    "usda": ("USD (ASCII)", True),
    "abc": ("Alembic", True),
    "x3d": ("X3D", True),
}

def get_available_formats() -> List[str]:
    """Get list of available export formats based on available tools"""
    formats = []
    for fmt, (desc, requires_blender) in SUPPORTED_FORMATS.items():
        if requires_blender and not BLENDER_PATH:
            continue
        formats.append(fmt)
    return formats

AVAILABLE_FORMATS = get_available_formats()
log.info(f"Available export formats: {AVAILABLE_FORMATS}")

# Create directories
for dir_path in [CHECKPOINTS_DIR, INPUT_DIR, OUTPUT_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        log.info(f"Created directory: {dir_path}")

# ============================================================================
# CHECKPOINT DISCOVERY
# ============================================================================

def get_checkpoint_list() -> List[str]:
    """Scan checkpoint directory and return list of available models."""
    checkpoints = []
    
    log.info(f"Scanning for checkpoints in: {CHECKPOINTS_DIR}")
    
    if not os.path.exists(CHECKPOINTS_DIR):
        log.warning(f"Checkpoints directory does not exist: {CHECKPOINTS_DIR}")
        return ["PLEASE_ADD_CHECKPOINT_FILES"]
    
    try:
        files = os.listdir(CHECKPOINTS_DIR)
        for f in files:
            if f.endswith(('.pth', '.pt', '.safetensors', '.ckpt')):
                full_path = os.path.join(CHECKPOINTS_DIR, f)
                size_mb = os.path.getsize(full_path) / (1024 * 1024)
                log.info(f"  Found checkpoint: {f} ({size_mb:.1f} MB)")
                checkpoints.append(f)
    except Exception as e:
        log.error(f"Error scanning checkpoints: {e}")
    
    if not checkpoints:
        log.warning("No checkpoint files found!")
        return ["NO_CHECKPOINTS_FOUND"]
    
    return sorted(checkpoints)

AVAILABLE_CHECKPOINTS = get_checkpoint_list()

# ============================================================================
# IMPORT DUST3R COMPONENTS
# ============================================================================

log.info("Importing Dust3r components...")

try:
    from dust3r.inference import inference as dust3r_inference
    log.info("  ✓ dust3r.inference")
except ImportError as e:
    log.error(f"  ✗ dust3r.inference: {e}")
    dust3r_inference = None

try:
    from dust3r.model import load_model, AsymmetricCroCo3DStereo
    log.info("  ✓ dust3r.model")
except ImportError as e:
    log.error(f"  ✗ dust3r.model: {e}")
    load_model = None

try:
    from dust3r.image_pairs import make_pairs
    log.info("  ✓ dust3r.image_pairs")
except ImportError as e:
    log.error(f"  ✗ dust3r.image_pairs: {e}")
    make_pairs = None

try:
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    log.info("  ✓ dust3r.cloud_opt")
except ImportError as e:
    log.error(f"  ✗ dust3r.cloud_opt: {e}")
    global_aligner = None

try:
    from dust3r.utils.device import to_numpy
    from dust3r.utils.image import load_images, rgb, ImgNorm
    log.info("  ✓ dust3r.utils")
except ImportError as e:
    log.error(f"  ✗ dust3r.utils: {e}")
    to_numpy = None

try:
    from dust3r.viz import pts3d_to_trimesh, cat_meshes, add_scene_cam, CAM_COLORS, OPENGL
    log.info("  ✓ dust3r.viz")
except ImportError as e:
    log.error(f"  ✗ dust3r.viz: {e}")

try:
    import trimesh
    log.info("  ✓ trimesh")
except ImportError as e:
    log.error(f"  ✗ trimesh: {e}")

try:
    from scipy.spatial.transform import Rotation
    log.info("  ✓ scipy")
except ImportError as e:
    log.error(f"  ✗ scipy: {e}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI tensor (B,H,W,C) to PIL Image"""
    if tensor.dim() == 4:
        tensor = tensor[0]
    np_img = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np_img)


def prepare_images_for_dust3r(images: torch.Tensor, size: int = 512) -> List[Dict]:
    """Convert ComfyUI image tensors to Dust3r format."""
    log.info(f"Preparing {len(images)} images (target size: {size})")
    
    dust3r_imgs = []
    
    for idx, img_tensor in enumerate(images):
        pil_img = tensor_to_pil(img_tensor.unsqueeze(0))
        W1, H1 = pil_img.size
        
        S = max(pil_img.size)
        interp = Image.LANCZOS if S > size else Image.BICUBIC
        new_size = tuple(int(round(x * size / S)) for x in pil_img.size)
        pil_img = pil_img.resize(new_size, interp)
        
        W, H = pil_img.size
        cx, cy = W // 2, H // 2
        
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if W == H:
            halfh = int(3 * halfw / 4)
        pil_img = pil_img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
        
        if idx % 10 == 0:
            log.debug(f"  Image {idx}: {W1}x{H1} -> {pil_img.size[0]}x{pil_img.size[1]}")
        
        img_tensor = ImgNorm(pil_img)[None]
        
        dust3r_imgs.append({
            'img': img_tensor,
            'true_shape': np.int32([pil_img.size[::-1]]),
            'idx': idx,
            'instance': str(idx)
        })
    
    log.info(f"Prepared {len(dust3r_imgs)} images")
    return dust3r_imgs


# ============================================================================
# BLENDER FORMAT CONVERSION
# ============================================================================

def convert_with_blender(input_path: str, output_path: str, output_format: str) -> bool:
    """Convert 3D file using Blender."""
    if not BLENDER_PATH:
        log.error("Blender not available")
        return False
    
    log.info(f"Converting to {output_format.upper()} using Blender...")
    
    script = f'''
import bpy
import sys

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath="{input_path}")

output_path = "{output_path}"
fmt = "{output_format}"

try:
    if fmt == "fbx":
        bpy.ops.export_scene.fbx(filepath=output_path, use_selection=False)
    elif fmt == "blend":
        bpy.ops.wm.save_as_mainfile(filepath=output_path)
    elif fmt in ["usd", "usdc", "usda"]:
        bpy.ops.wm.usd_export(filepath=output_path)
    elif fmt == "abc":
        bpy.ops.wm.alembic_export(filepath=output_path)
    elif fmt == "x3d":
        bpy.ops.export_scene.x3d(filepath=output_path)
    elif fmt == "obj":
        bpy.ops.wm.obj_export(filepath=output_path)
    elif fmt == "ply":
        bpy.ops.wm.ply_export(filepath=output_path)
    elif fmt == "stl":
        bpy.ops.export_mesh.stl(filepath=output_path)
    elif fmt == "dae":
        bpy.ops.wm.collada_export(filepath=output_path)
    else:
        print(f"Unsupported: {{fmt}}")
        sys.exit(1)
    print(f"Exported: {{output_path}}")
except Exception as e:
    print(f"Error: {{e}}")
    sys.exit(1)
'''
    
    script_path = os.path.join(OUTPUT_DIR, "blender_convert.py")
    with open(script_path, 'w') as f:
        f.write(script)
    
    try:
        result = subprocess.run(
            [BLENDER_PATH, "--background", "--python", script_path],
            capture_output=True, text=True, timeout=120
        )
        
        if result.returncode != 0:
            log.error(f"Blender failed: {result.stderr}")
            return False
        
        log.info("✓ Blender conversion successful")
        return True
        
    except subprocess.TimeoutExpired:
        log.error("Blender timed out")
        return False
    except Exception as e:
        log.error(f"Blender error: {e}")
        return False
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)


def convert_scene_to_3d(
    outdir: str,
    imgs: List,
    pts3d: np.ndarray,
    mask: np.ndarray,
    focals: np.ndarray,
    cams2world: np.ndarray,
    output_format: str = "glb",
    cam_size: float = 0.05,
    as_pointcloud: bool = False,
    transparent_cams: bool = False
) -> str:
    """Export scene to 3D file in specified format"""
    
    log.info(f"Exporting to {output_format.upper()}...")
    
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)
    
    scene = trimesh.Scene()
    
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)
    
    for i, pose_c2w in enumerate(cams2world):
        cam_color = CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(
            scene, pose_c2w, cam_color,
            None if transparent_cams else imgs[i],
            focals[i],
            imsize=imgs[i].shape[1::-1],
            screen_width=cam_size
        )
    
    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    
    base_name = "scene"
    glb_path = os.path.join(outdir, f'{base_name}.glb')
    final_path = os.path.join(outdir, f'{base_name}.{output_format}')
    
    requires_blender = SUPPORTED_FORMATS.get(output_format, (None, False))[1]
    
    if output_format == "glb":
        scene.export(file_obj=glb_path)
        log.info(f"  Exported: {glb_path}")
        return glb_path
    
    elif not requires_blender:
        try:
            scene.export(file_obj=final_path)
            log.info(f"  Exported: {final_path}")
            return final_path
        except Exception as e:
            log.warning(f"  Direct export failed: {e}")
    
    # Export GLB then convert with Blender
    scene.export(file_obj=glb_path)
    
    if BLENDER_PATH:
        if convert_with_blender(glb_path, final_path, output_format):
            if os.path.exists(final_path):
                log.info(f"  Exported: {final_path}")
                return final_path
    
    log.warning("  Returning GLB instead")
    return glb_path


# ============================================================================
# COMFYUI NODES
# ============================================================================

class Dust3rModelLoader:
    """Load a Dust3r model from checkpoint file."""
    
    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = get_checkpoint_list()
        return {
            "required": {
                "checkpoint": (checkpoints, {"default": checkpoints[0] if checkpoints else "NO_CHECKPOINTS_FOUND"}),
                "device": (["cuda", "cpu", "auto"], {"default": "auto"}),
            },
        }
    
    RETURN_TYPES = ("DUST3R_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Dust3r"
    
    def load_model(self, checkpoint: str, device: str):
        log.info("=" * 60)
        log.info("DUST3R MODEL LOADER")
        log.info("=" * 60)
        
        if checkpoint in ["NO_CHECKPOINTS_FOUND", "PLEASE_ADD_CHECKPOINT_FILES"]:
            raise ValueError(f"No checkpoints found in {CHECKPOINTS_DIR}")
        
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, checkpoint)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        log.info(f"Checkpoint: {checkpoint}")
        log.info(f"Device: {device}")
        mem_manager.log_memory_status()
        mem_manager.clear_all_memory()
        
        try:
            model = load_model(checkpoint_path, device, verbose=True)
            log.info("✓ Model loaded!")
            return (model,)
        except Exception as e:
            log.error(f"Failed: {e}")
            traceback.print_exc()
            raise


class Dust3rReconstruct:
    """Run Dust3r 3D reconstruction with memory optimization."""
    
    @classmethod
    def INPUT_TYPES(cls):
        formats = get_available_formats()
        return {
            "required": {
                "model": ("DUST3R_MODEL",),
                "images": ("IMAGE",),
                "output_format": (formats, {"default": "glb"}),
                "image_size": ("INT", {"default": 512, "min": 224, "max": 1024, "step": 16}),
                "scene_graph": (["complete", "swin", "oneref"], {"default": "complete"}),
                "schedule": (["linear", "cosine"], {"default": "linear"}),
                "iterations": ("INT", {"default": 300, "min": 0, "max": 5000, "step": 50}),
                "min_conf_threshold": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "cam_size": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 0.5, "step": 0.01}),
                "as_pointcloud": ("BOOLEAN", {"default": False}),
                "mask_sky": ("BOOLEAN", {"default": False}),
                "clean_depth": ("BOOLEAN", {"default": True}),
                "transparent_cams": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "device": (["cuda", "cpu", "auto"], {"default": "auto"}),
                "memory_mode": (["auto", "low_memory", "high_memory"], {"default": "auto"}),
                "max_images": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "window_size": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("output_path", "depth_maps", "confidence_maps")
    FUNCTION = "reconstruct"
    CATEGORY = "Dust3r"
    OUTPUT_NODE = True
    
    def reconstruct(
        self, model, images: torch.Tensor,
        output_format: str = "glb", image_size: int = 512,
        scene_graph: str = "complete", schedule: str = "linear",
        iterations: int = 300, min_conf_threshold: float = 3.0,
        cam_size: float = 0.05, as_pointcloud: bool = False,
        mask_sky: bool = False, clean_depth: bool = True,
        transparent_cams: bool = True, device: str = "auto",
        memory_mode: str = "auto", max_images: int = 0,
        window_size: int = 3
    ):
        log.info("=" * 60)
        log.info("DUST3R RECONSTRUCTION")
        log.info("=" * 60)
        
        num_images = len(images)
        log.info(f"Total input images: {num_images}")
        log.info(f"Output format: {output_format.upper()}")
        log.info(f"Image size: {image_size}")
        log.info(f"Scene graph: {scene_graph}")
        log.info(f"Memory mode: {memory_mode}")
        
        # Limit images if max_images is set
        if max_images > 0 and num_images > max_images:
            log.warning(f"Limiting to {max_images} images (from {num_images})")
            # Sample evenly across the sequence
            indices = np.linspace(0, num_images - 1, max_images, dtype=int)
            images = images[indices]
            num_images = max_images
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Device: {device}")
        
        mem_manager.log_memory_status()
        gpu_info = mem_manager.get_gpu_memory_info()
        
        # Auto-select memory mode and adjust parameters
        if memory_mode == "auto":
            if num_images > 30:
                memory_mode = "low_memory"
                log.warning(f"Large sequence ({num_images} images) - using low_memory mode")
            elif num_images > 15 or gpu_info['free'] < 6:
                memory_mode = "low_memory"
            else:
                memory_mode = "high_memory"
            log.info(f"Auto memory mode: {memory_mode}")
        
        # For large sequences, force swin scene graph to limit pairs
        if num_images > 20 and scene_graph == "complete":
            log.warning(f"Complete graph with {num_images} images creates {num_images*(num_images-1)//2} pairs!")
            log.warning(f"Switching to 'swin' scene graph to reduce memory usage")
            scene_graph = "swin"
        
        # Calculate expected pairs and warn
        if scene_graph == "complete":
            expected_pairs = num_images * (num_images - 1)  # symmetrized
        elif scene_graph == "swin":
            expected_pairs = num_images * window_size * 2  # symmetrized
        else:  # oneref
            expected_pairs = (num_images - 1) * 2
        
        log.info(f"Expected image pairs: ~{expected_pairs}")
        
        if expected_pairs > 500:
            log.warning(f"Large number of pairs ({expected_pairs}) - this will be slow!")
            log.warning("Consider: reducing images, using 'swin' graph, or setting max_images")
        
        batch_size = 1 if memory_mode == "low_memory" else mem_manager.estimate_batch_size(num_images, image_size, gpu_info['free'])
        log.info(f"Inference batch size: {batch_size}")
        
        try:
            mem_manager.clear_all_memory()
            
            # Step 1: Prepare images
            log.info("-" * 40)
            log.info("Step 1: Preparing images...")
            dust3r_imgs = prepare_images_for_dust3r(images, size=image_size)
            
            # Free original images tensor
            del images
            mem_manager.clear_gpu_memory()
            
            if len(dust3r_imgs) == 1:
                import copy
                dust3r_imgs = [dust3r_imgs[0], copy.deepcopy(dust3r_imgs[0])]
                dust3r_imgs[1]['idx'] = 1
            
            # Step 2: Create pairs
            log.info("-" * 40)
            log.info("Step 2: Creating pairs...")
            
            scene_graph_str = scene_graph
            if scene_graph == "swin":
                # Use provided window_size or calculate
                winsize = min(window_size, max(1, (len(dust3r_imgs) - 1) // 2))
                scene_graph_str = f"swin-{winsize}"
                log.info(f"Using sliding window size: {winsize}")
            elif scene_graph == "oneref":
                scene_graph_str = "oneref-0"
            
            pairs = make_pairs(dust3r_imgs, scene_graph=scene_graph_str, prefilter=None, symmetrize=True)
            actual_pairs = len(pairs)
            log.info(f"Created {actual_pairs} pairs")
            
            if memory_mode == "low_memory":
                mem_manager.clear_gpu_memory()
            
            # Step 3: Inference (this is what takes time with many pairs)
            log.info("-" * 40)
            log.info(f"Step 3: Running inference on {actual_pairs} pairs...")
            log.info(f"  This may take a while for large sequences...")
            mem_manager.log_memory_status()
            
            output = dust3r_inference(pairs, model, device, batch_size=batch_size)
            log.info("✓ Inference complete")
            
            # Free pairs
            del pairs
            mem_manager.clear_gpu_memory()
            
            # Step 4: Alignment
            log.info("-" * 40)
            log.info("Step 4: Global alignment...")
            
            n_imgs = len(dust3r_imgs)
            log.info(f"Number of images for alignment: {n_imgs}")
            
            # Use PointCloudOptimizer for 3+ images, PairViewer for 2
            if n_imgs > 2:
                mode = GlobalAlignerMode.PointCloudOptimizer
                log.info("Using PointCloudOptimizer mode")
            else:
                mode = GlobalAlignerMode.PairViewer
                log.info("Using PairViewer mode (2 images only)")
            
            # Ensure we're on the correct device
            log.info(f"Creating scene on device: {device}")
            scene = global_aligner(output, device=device, mode=mode)
            
            # Log scene info
            log.info(f"Scene created with {scene.n_imgs} images, {len(scene.edges)} edges")
            
            # Free inference output
            del output
            mem_manager.clear_gpu_memory()
            
            # Only run optimization for PointCloudOptimizer mode with 3+ images
            if mode == GlobalAlignerMode.PointCloudOptimizer and n_imgs > 2:
                # Adjust iterations based on memory mode and sequence length
                if memory_mode == "low_memory":
                    actual_iters = min(iterations, 150)
                elif n_imgs > 30:
                    actual_iters = min(iterations, 200)
                else:
                    actual_iters = iterations
                
                # Ensure scene is in training mode for gradients
                scene.train()
                
                # Check if there are parameters to optimize
                params_to_optimize = [p for p in scene.parameters() if p.requires_grad]
                log.info(f"Parameters requiring gradients: {len(params_to_optimize)}")
                
                # Log parameter info for debugging
                for name, param in scene.named_parameters():
                    if param.requires_grad:
                        log.debug(f"  Trainable: {name} - shape {param.shape}")
                
                if len(params_to_optimize) == 0:
                    log.warning("No parameters require gradients - attempting to enable gradients")
                    # Try to enable gradients on key parameters
                    if hasattr(scene, 'im_depthmaps'):
                        scene.im_depthmaps.requires_grad_(True)
                    if hasattr(scene, 'im_poses'):
                        scene.im_poses.requires_grad_(True)
                    if hasattr(scene, 'im_focals'):
                        scene.im_focals.requires_grad_(True)
                    params_to_optimize = [p for p in scene.parameters() if p.requires_grad]
                    log.info(f"After enabling: {len(params_to_optimize)} parameters require gradients")
                
                if len(params_to_optimize) == 0:
                    log.warning("Still no trainable parameters - skipping optimization")
                else:
                    log.info(f"Optimizing for {actual_iters} iterations...")
                    
                    try:
                        # Run optimization
                        loss = scene.compute_global_alignment(init='mst', niter=actual_iters, schedule=schedule, lr=0.01)
                        log.info(f"Final loss: {loss}")
                    except RuntimeError as e:
                        error_msg = str(e)
                        if "does not require grad" in error_msg:
                            log.warning(f"Optimization skipped (gradient error): {error_msg}")
                            log.warning("Continuing without optimization - results may be less accurate")
                        else:
                            log.error(f"Optimization error: {error_msg}")
                            raise
            else:
                log.info("Skipping optimization (PairViewer mode or insufficient images)")
            
            mem_manager.clear_gpu_memory()
            
            # Step 5: Post-process
            log.info("-" * 40)
            log.info("Step 5: Post-processing...")
            
            if clean_depth:
                scene = scene.clean_pointcloud()
            if mask_sky:
                scene = scene.mask_sky()
            
            # Step 6: Extract
            log.info("-" * 40)
            log.info("Step 6: Extracting results...")
            
            rgbimg = scene.imgs
            focals = scene.get_focals().cpu()
            cams2world = scene.get_im_poses().cpu()
            pts3d = to_numpy(scene.get_pts3d())
            scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_threshold)))
            masks = to_numpy(scene.get_masks())
            
            # Step 7: Export
            log.info("-" * 40)
            log.info("Step 7: Exporting...")
            
            output_path = convert_scene_to_3d(
                OUTPUT_DIR, rgbimg, pts3d, masks, focals, cams2world,
                output_format=output_format, cam_size=cam_size,
                as_pointcloud=as_pointcloud, transparent_cams=transparent_cams
            )
            
            # Step 8: Visualizations
            log.info("-" * 40)
            log.info("Step 8: Creating visualizations...")
            
            import matplotlib.pyplot as plt
            
            depths = to_numpy(scene.get_depthmaps())
            confs = to_numpy([c for c in scene.im_conf])
            
            depths_max = max([d.max() for d in depths])
            depths_norm = [d / depths_max for d in depths]
            
            cmap = plt.get_cmap('jet')
            confs_max = max([c.max() for c in confs])
            confs_colored = [cmap(c / confs_max)[..., :3] for c in confs]
            
            depth_tensors = [torch.from_numpy(np.stack([d, d, d], axis=-1)).float() for d in depths_norm]
            conf_tensors = [torch.from_numpy(c).float() for c in confs_colored]
            
            depth_batch = torch.stack(depth_tensors, dim=0)
            conf_batch = torch.stack(conf_tensors, dim=0)
            
            # Final cleanup
            del scene, rgbimg, pts3d, masks, depths, confs
            mem_manager.clear_all_memory()
            
            log.info("=" * 60)
            log.info(f"COMPLETE! Output: {output_path}")
            log.info("=" * 60)
            
            return (output_path, depth_batch, conf_batch)
            
        except Exception as e:
            log.error(f"Failed: {e}")
            traceback.print_exc()
            mem_manager.clear_all_memory()
            raise


class Dust3rSimple:
    """Simplified all-in-one Dust3r node."""
    
    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = get_checkpoint_list()
        formats = get_available_formats()
        return {
            "required": {
                "images": ("IMAGE",),
                "checkpoint": (checkpoints, {"default": checkpoints[0] if checkpoints else "NO_CHECKPOINTS_FOUND"}),
                "output_format": (formats, {"default": "glb"}),
            },
            "optional": {
                "image_size": ("INT", {"default": 512, "min": 224, "max": 1024, "step": 16}),
                "iterations": ("INT", {"default": 300, "min": 0, "max": 2000, "step": 50}),
                "memory_mode": (["auto", "low_memory", "high_memory"], {"default": "auto"}),
                "max_images": ("INT", {"default": 0, "min": 0, "max": 200, "step": 1}),
                "scene_graph": (["complete", "swin", "oneref"], {"default": "swin"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "run"
    CATEGORY = "Dust3r"
    OUTPUT_NODE = True
    
    def run(self, images, checkpoint, output_format="glb", image_size=512, 
            iterations=300, memory_mode="auto", max_images=0, scene_graph="swin"):
        loader = Dust3rModelLoader()
        model, = loader.load_model(checkpoint, "auto")
        
        reconstructor = Dust3rReconstruct()
        output_path, _, _ = reconstructor.reconstruct(
            model=model, images=images, output_format=output_format,
            image_size=image_size, iterations=iterations, memory_mode=memory_mode,
            max_images=max_images, scene_graph=scene_graph
        )
        return (output_path,)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "Dust3rModelLoader": Dust3rModelLoader,
    "Dust3rReconstruct": Dust3rReconstruct,
    "Dust3rSimple": Dust3rSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Dust3rModelLoader": "Dust3r Model Loader",
    "Dust3rReconstruct": "Dust3r Reconstruct 3D",
    "Dust3rSimple": "Dust3r Simple (All-in-One)",
}

log.info("=" * 60)
log.info("DUST3R NODES REGISTERED")
log.info(f"  Formats: {AVAILABLE_FORMATS}")
log.info(f"  Blender: {'Yes' if BLENDER_PATH else 'No'}")
log.info("=" * 60)
