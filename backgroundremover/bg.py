import io
import os
import typing
from PIL import Image, ImageOps
# Pillow>=10 removed Image.ANTIALIAS; moviepy 1.x still references it during resize.
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from scipy.ndimage.morphology import binary_erosion
try:
    from moviepy import VideoFileClip
except ImportError:  # moviepy 1.x exposes it under moviepy.editor
    from moviepy.editor import VideoFileClip
import numpy as np
import torch
import torch.nn.functional
import torch.nn.functional
from hsh.library.hash import Hasher
from .u2net import detect, u2net
from . import github

# Register HEIC format support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # HEIC support is optional

# closes https://github.com/nadermx/backgroundremover/issues/18
# closes https://github.com/nadermx/backgroundremover/issues/112
try:
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
        _gpu_name = torch.cuda.get_device_name(0)
        _gpu_mem = torch.cuda.get_device_properties(0).total_memory
        print(f"Device: CUDA ({_gpu_name}, {_gpu_mem // (1024**2)}MB VRAM)")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
        print("Device: MPS (Apple Silicon GPU)")
    else:
        DEVICE = torch.device('cpu')
        print("Device: CPU (no GPU detected - install CUDA toolkit for GPU acceleration)")
except Exception as e:
    print(f"Device: CPU (Setting CUDA or MPS failed: {e})")
    DEVICE = torch.device('cpu')


def max_workers(model_name="u2net", gpu_batchsize=2):
    """Estimate max safe worker processes based on available GPU/system memory.

    Each worker spawns a separate process that loads its own copy of the model
    plus a CUDA context. This estimates how many can fit in VRAM.
    """
    if torch.cuda.is_available():
        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory
        except Exception:
            return 1

        # Per-worker VRAM estimate:
        #   CUDA context per process:  ~400MB
        #   Model weights (float32):   ~175MB (u2net/human_seg), ~5MB (u2netp)
        #   JIT traced copy:           same as model weights
        #   Batch inference tensors:   ~30MB per frame in batch
        if model_name == "u2netp":
            model_bytes = 5 * 1024 * 1024
        else:
            model_bytes = 175 * 1024 * 1024

        per_worker = (
            400 * 1024 * 1024       # CUDA context overhead
            + model_bytes * 2       # model + JIT trace
            + gpu_batchsize * 30 * 1024 * 1024  # inference tensors
        )

        # Reserve 512MB for OS/driver/display
        usable = total_mem - 512 * 1024 * 1024
        calculated = max(1, int(usable // per_worker))
        return calculated

    # CPU/MPS: limit by CPU cores (inference is compute-bound)
    cpu_count = os.cpu_count() or 2
    return max(1, cpu_count // 2)

class Net(torch.nn.Module):
    def __init__(self, model_name):
        super(Net, self).__init__()
        hasher = Hasher()
        model = {
            'u2netp': (u2net.U2NETP,
                       'e4f636406ca4e2af789941e7f139ee2e',
                       '1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy',
                       'U2NET_PATH'),
            'u2net': (u2net.U2NET,
                      '09fb4e49b7f785c9f855baf94916840a',
                      '1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ',
                      'U2NET_PATH'),
            'u2net_human_seg': (u2net.U2NET,
                                '347c3d51b01528e5c6c071e3cff1cb55',
                                '1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P',
                                'U2NET_PATH')
        }[model_name]

        if model_name == "u2netp":
            net = u2net.U2NETP(3, 1)
            path = os.environ.get(
                "U2NETP_PATH",
                os.path.expanduser(os.path.join("~", ".u2net", model_name + ".pth")),
            )
            if (
                not os.path.exists(path)
            ):
                github.download_files_from_github(
                    path, model_name
                )

        elif model_name == "u2net":
            net = u2net.U2NET(3, 1)
            path = os.environ.get(
                "U2NET_PATH",
                os.path.expanduser(os.path.join("~", ".u2net", model_name + ".pth")),
            )
            if (
                not os.path.exists(path)
                #or hasher.md5(path) != "09fb4e49b7f785c9f855baf94916840a"
            ):
                github.download_files_from_github(
                    path, model_name
                )

        elif model_name == "u2net_human_seg":
            net = u2net.U2NET(3, 1)
            path = os.environ.get(
                "U2NET_PATH",
                os.path.expanduser(os.path.join("~", ".u2net", model_name + ".pth")),
            )
            if (
                not os.path.exists(path)
                #or hasher.md5(path) != "347c3d51b01528e5c6c071e3cff1cb55"
            ):
                github.download_files_from_github(
                    path, model_name
                )
        else:
            print("Choose between u2net, u2net_human_seg or u2netp", file=sys.stderr)

        try:
            net.load_state_dict(torch.load(path, map_location=torch.device(DEVICE)))
            net.to(device=DEVICE, dtype=torch.float32, non_blocking=True)
            net.eval()
            self.net = net
        except EOFError:
            print(f"\n{'='*60}")
            print(f"ERROR: Model file appears to be corrupted or incomplete!")
            print(f"Path: {path}")
            print(f"\nThis usually happens when the model download was interrupted.")
            print(f"To fix this:")
            print(f"  1. Delete the corrupted file: rm {path}")
            print(f"  2. Run backgroundremover again to re-download the model")
            print(f"{'='*60}\n")
            raise RuntimeError(f"Corrupted model file at {path}. Please delete it and re-run to download again.")
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"ERROR: Failed to load model '{model_name}'")
            print(f"Path: {path}")
            print(f"Error: {e}")
            print(f"\nIf the error persists:")
            print(f"  1. Try deleting the model file: rm {path}")
            print(f"  2. Run backgroundremover again to re-download")
            print(f"  3. Check if you have enough disk space")
            print(f"{'='*60}\n")
            raise

    def forward(self, block_input: torch.Tensor):
        image_data = block_input.permute(0, 3, 1, 2)
        original_shape = image_data.shape[2:]
        image_data = torch.nn.functional.interpolate(image_data, (320, 320), mode='bilinear')
        image_data = (image_data / 255 - 0.485) / 0.229
        out = self.net(image_data)[0][:, 0:1]
        ma = torch.max(out)
        mi = torch.min(out)
        out = (out - mi) / (ma - mi) * 255
        out = torch.nn.functional.interpolate(out, original_shape, mode='bilinear')
        out = out[:, 0]
        out = out.to(dtype=torch.uint8, device=torch.device('cpu'), non_blocking=True).detach()
        return out


def alpha_matting_cutout(
    img,
    mask,
    foreground_threshold,
    background_threshold,
    erode_structure_size,
    base_size,
):
    size = img.size

    img.thumbnail((base_size, base_size), Image.LANCZOS)
    mask = mask.resize(img.size, Image.LANCZOS)

    img = np.asarray(img)
    mask = np.asarray(mask)

    # guess likely foreground/background
    is_foreground = mask > foreground_threshold
    is_background = mask < background_threshold

    # erode foreground/background
    structure = None
    if erode_structure_size > 0:
        structure = np.ones((erode_structure_size, erode_structure_size), dtype=np.int64)

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    # build trimap
    # 0   = background
    # 128 = unknown
    # 255 = foreground
    trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    # build the cutout image
    img_normalized = img / 255.0
    trimap_normalized = trimap / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)

    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
    cutout = Image.fromarray(cutout)
    cutout = cutout.resize(size, Image.LANCZOS)

    return cutout


def naive_cutout(img, mask):
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask.resize(img.size, Image.LANCZOS))
    return cutout


def get_model(model_name):
    if model_name == "u2netp":
        return detect.load_model(model_name="u2netp")
    if model_name == "u2net_human_seg":
        return detect.load_model(model_name="u2net_human_seg")
    else:
        return detect.load_model(model_name="u2net")


def remove(
    data,
    model_name="u2net",
    alpha_matting=False,
    alpha_matting_foreground_threshold=240,
    alpha_matting_background_threshold=10,
    alpha_matting_erode_structure_size=10,
    alpha_matting_base_size=1000,
    only_mask=False,
    background_color=None,
    background_image=None,
    mask_threshold=None,
):
    model = get_model(model_name)

    if isinstance(data, np.ndarray):
        img = Image.fromarray(data).convert("RGB")
    else:
        try:
            img = Image.open(io.BytesIO(data))
            # Handle EXIF orientation to prevent rotated images (fixes #144)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image input to `remove()`: {e}")

    mask = detect.predict(model, np.array(img)).convert("L")

    # Apply threshold for hard/sharp edges (fixes #122)
    if mask_threshold is not None:
        mask = mask.point(lambda p: 255 if p > mask_threshold else 0)

    # If only_mask is True, return just the mask
    if only_mask:
        bio = io.BytesIO()
        mask.save(bio, "PNG")
        return bio.getbuffer()

    if alpha_matting:
        cutout = alpha_matting_cutout(
            img,
            mask,
            alpha_matting_foreground_threshold,
            alpha_matting_background_threshold,
            alpha_matting_erode_structure_size,
            alpha_matting_base_size,
        )
    else:
        cutout = naive_cutout(img, mask)

    # If background_image is specified, composite over that image
    if background_image is not None:
        if isinstance(background_image, np.ndarray):
            bg = Image.fromarray(background_image).convert("RGB")
        else:
            try:
                bg = Image.open(io.BytesIO(background_image))
                # Handle EXIF orientation for background image too
                bg = ImageOps.exif_transpose(bg)
                bg = bg.convert("RGB")
            except Exception as e:
                raise ValueError(f"Invalid background image input: {e}")

        # Resize background to match cutout size
        bg = bg.resize(cutout.size, Image.LANCZOS)

        if cutout.mode == 'RGBA':
            bg.paste(cutout, mask=cutout.split()[3])
            cutout = bg
        else:
            cutout = bg
    # If background_color is specified, composite with that color
    elif background_color is not None:
        bg = Image.new("RGB", cutout.size, background_color)
        if cutout.mode == 'RGBA':
            bg.paste(cutout, mask=cutout.split()[3])
            cutout = bg
        else:
            cutout = bg

    bio = io.BytesIO()
    cutout.save(bio, "PNG")

    return bio.getbuffer()


def iter_frames(path):
    clip = VideoFileClip(path)
    # moviepy 2.x renamed Clip.resize() -> Clip.resized(); support both.
    resizer = getattr(clip, "resized", None) or getattr(clip, "resize", None)
    if resizer is not None:
        clip = resizer(height=320)
    return clip.iter_frames(dtype="uint8")


@torch.no_grad()
def remove_many(image_data: typing.List[np.array], net: Net):
    image_data = np.stack(image_data)
    image_data = torch.as_tensor(image_data, dtype=torch.float32, device=DEVICE)
    return net(image_data).numpy()
