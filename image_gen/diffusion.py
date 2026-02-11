"""
Local diffusion image generation. Requires PyTorch + diffusers; see requirements-image-gen.txt.
Run with your venv so CUDA is used if available:  venv\\Scripts\\activate  then  python image_gen/diffusion.py
If you see 'DLL load failed while importing _C': install Visual C++ Redistributable
(https://aka.ms/vs/17/release/vc_redist.x64.exe) and reinstall PyTorch from pytorch.org.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import torch
from diffusers import ZImagePipeline, ZImageTransformer2DModel, GGUFQuantizationConfig

# -------------------------------
# Config
# -------------------------------
_IMAGE_GEN_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_IMAGE_GEN_DIR, "z_image_turbo-Q4_K_M.gguf")
TOKENIZER_PATH = os.path.join(_IMAGE_GEN_DIR, "config.json")
WIDTH, HEIGHT = 512, 512 # image size (must be divisible by pipeline's vae_scale_factor * 2)
STEPS = 9
GUIDANCE_SCALE = 0.0  # typical for turbo
GENERATION_TIMEOUT_SECONDS = 600  # raise TimeoutError if pipeline takes longer

# -------------------------------
# CUDA check
# -------------------------------
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available. Activate your venv with CUDA torch and run with GPU.")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

def fast_generate(prompt, save_path, height=HEIGHT, width=WIDTH):
    print("Loading Z-Image pipeline and local transformer...")
    t0 = time.perf_counter()
    transformer = ZImageTransformer2DModel.from_single_file(
        MODEL_PATH,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16,
        device="cuda"
    )

    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        transformer=transformer,
    )

    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.vae.to("cpu")
    pipe.safety_checker = None

    print(f"Pipeline loaded in {time.perf_counter() - t0:.2f}s")

    print("Generating image...")
    t1 = time.perf_counter()

    def _run():
        return pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE_SCALE,
        ).images[0]

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run)
        try:
            image = future.result(timeout=GENERATION_TIMEOUT_SECONDS)
        except FuturesTimeoutError:
            raise TimeoutError(f"Image generation did not finish within {GENERATION_TIMEOUT_SECONDS}s")

    elapsed = time.perf_counter() - t1
    print(f"Image generation took {elapsed:.2f}s")
    image.save(save_path)
    return image