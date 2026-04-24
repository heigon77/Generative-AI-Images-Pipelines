import torch
from diffusers import FluxPipeline
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = "black-forest-labs/FLUX.1-schnell"  # schnell is faster & lighter
                                                   # swap for FLUX.1-dev if you have HF access
PROMPT     = "An woman furry fox warrior, full body, cinematic lighting, ultra detailed"
NEG_PROMPT = ""           # Flux doesn't use negative prompts natively
WIDTH      = 832           # keep ≤768 on 8GB VRAM
HEIGHT     = 1216
STEPS      = 4             # schnell needs only 1-4 steps; dev needs ~20-50
GUIDANCE   = 0.0           # schnell ignores CFG; set to 3.5+ for dev
SEED       = None          # set to an int for reproducibility, or None for random
OUTPUT     = "Outputs/Flux/output.png"
# ─────────────────────────────────────────────────────────────────────────────


def load_pipeline(model_id: str) -> FluxPipeline:
    print(f"Loading {model_id} ...")

    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,   # bfloat16 saves ~40% VRAM vs float32
    )

    # Sequential CPU offload: moves layers to GPU only when needed.
    # Slower than full-GPU but keeps peak VRAM under 8 GB.
    pipe.enable_sequential_cpu_offload()

    # Optional extras if you still hit OOM:
    # pipe.enable_attention_slicing()        # slices attention to save VRAM
    # pipe.vae.enable_tiling()               # tiles VAE decode for large images

    return pipe


def generate(pipe: FluxPipeline) -> None:
    generator = torch.Generator(device="cuda").manual_seed(SEED) if SEED is not None else None

    print(f"Generating {WIDTH}×{HEIGHT} image ({STEPS} steps) ...")
    result = pipe(
        prompt=PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=generator,
    )

    image = result.images[0]
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT)
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    pipe = load_pipeline(MODEL_ID)
    generate(pipe)