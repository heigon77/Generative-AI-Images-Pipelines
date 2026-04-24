import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, EulerAncestralDiscreteScheduler
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
model_path  = "Models/ponyDiffusionV6XL_v6StartWithThisOne.safetensors"
vae_path    = "Models/sdxl_vae.safetensors"
lora_path   = "Models/HandFixer_pdxl_Incrs_v1.safetensors"
output_path = "Outputs/Pony/output.png"

PROMPT = "score_9, score_8_up, score_7_up, 1girl, furry, fox, woman, fully dressed, (highly detailed hands), (perfect fingers), (fine motor skills), masterpiece"
NEGATIVE_PROMPT = "score_4, score_5, score_6, source_pony, low quality, bad anatomy, rating_explicit, NSFW, (bad hands, broken fingers, extra fingers, fewer digits, fused fingers, missing fingers, deformed fingers, malformed hands:1.2), (intertwined fingers:1.1)"
WIDTH      = 832
HEIGHT     = 1216
STEPS      = 30
GUIDANCE   = 7.0
SEED       = None
LORA_SCALE = 0.75
# ─────────────────────────────────────────────────────────────────────────────


def load_vae(vae_path: str) -> AutoencoderKL:
    print("Loading VAE...")
    vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float16)
    return vae


def load_pipeline(model_path: str, vae: AutoencoderKL) -> StableDiffusionXLPipeline:
    print("Loading Base Model (Pony)... This may take a while.")
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    return pipe


def apply_lora(pipe: StableDiffusionXLPipeline, lora_path: str, lora_scale: float) -> StableDiffusionXLPipeline:
    print(f"Loading HandFixer LoRA: {lora_path} ...")
    pipe.load_lora_weights(lora_path, adapter_name="handfix")

    print("Fusing LoRA into the model...")
    pipe.fuse_lora(lora_scale=lora_scale)
    return pipe


def configure_pipeline(pipe: StableDiffusionXLPipeline) -> StableDiffusionXLPipeline:
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    return pipe


def generate_image(pipe: StableDiffusionXLPipeline, prompt: str, negative_prompt: str,
                   width: int, height: int, steps: int, guidance: float, seed: int):
    print(f"Generating {width}×{height} image ({steps} steps)...")
    generator = torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator,
    ).images[0]
    return image


def save_image(image, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Success! Image saved → {output_path}")

if __name__ == "__main__":
    vae  = load_vae(vae_path)
    pipe = load_pipeline(model_path, vae)
    pipe = apply_lora(pipe, lora_path, LORA_SCALE)
    pipe = configure_pipeline(pipe)

    image = generate_image(pipe, PROMPT, NEGATIVE_PROMPT, WIDTH, HEIGHT, STEPS, GUIDANCE, SEED)
    save_image(image, output_path)
