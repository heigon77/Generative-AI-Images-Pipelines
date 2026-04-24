import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, EulerAncestralDiscreteScheduler
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
model_id    = "runwayml/stable-diffusion-v1-5"          # ID of HuggingFace
vae_id      = "stabilityai/sd-vae-ft-mse"               # None to use default VAE; set to a custom one if you want
lora_path   = None  # Local loRA file (ex: "Models/handfixer.safetensors") or None without loRA
output_path = "Outputs/SD15/output.png"

PROMPT          = "1girl, fantasy, detailed face, beautiful eyes, intricate armor, masterpiece, best quality"
NEGATIVE_PROMPT = "low quality, worst quality, bad anatomy, bad hands, text, watermark, blurry, deformed"
WIDTH      = 512
HEIGHT     = 512
STEPS      = 25
GUIDANCE   = 7.0
SEED       = None  # None for random seed, or set to an int for reproducibility
LORA_SCALE = 0.8
# ─────────────────────────────────────────────────────────────────────────────


def load_vae(vae_id: str) -> AutoencoderKL:
    print(f"Loading VAE from HuggingFace: {vae_id} ...")
    vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16)
    return vae


def load_pipeline(model_id: str, vae: AutoencoderKL | None) -> StableDiffusionPipeline:
    print(f"Loading Base Model from HuggingFace: {model_id} ...")
    print("(O modelo será baixado e cacheado automaticamente na primeira execução)")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    return pipe


def apply_lora(pipe: StableDiffusionPipeline, lora_path: str, lora_scale: float) -> StableDiffusionPipeline:
    print(f"Loading LoRA: {lora_path} ...")
    pipe.load_lora_weights(lora_path, adapter_name="custom_lora")

    print("Fusing LoRA into the model...")
    pipe.fuse_lora(lora_scale=lora_scale)
    return pipe


def configure_pipeline(pipe: StableDiffusionPipeline) -> StableDiffusionPipeline:
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    return pipe


def generate_image(pipe: StableDiffusionPipeline, prompt: str, negative_prompt: str,
                   width: int, height: int, steps: int, guidance: float, seed: int | None):
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


def main():
    vae  = load_vae(vae_id) if vae_id else None
    pipe = load_pipeline(model_id, vae)

    if lora_path:
        pipe = apply_lora(pipe, lora_path, LORA_SCALE)

    pipe  = configure_pipeline(pipe)
    image = generate_image(pipe, PROMPT, NEGATIVE_PROMPT, WIDTH, HEIGHT, STEPS, GUIDANCE, SEED)
    save_image(image, output_path)


if __name__ == "__main__":
    main()