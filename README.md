# 🎨 Local Image Generation Scripts

A collection of Python scripts for local AI image generation using three different pipelines: **FLUX.1-schnell**, **Pony Diffusion XL**, and **Stable Diffusion 1.5**. All scripts run on a local CUDA-enabled GPU via Diffusers.

---

## 📁 Project Structure

```
.
├── requirements.txt
├── ImageGenerator 
│   ├── flux.py       # FLUX.1-schnell (or FLUX.1-dev)
│   ├── pony.py       # Pony Diffusion XL + custom VAE + LoRA
│   ├── sd15.py       # Stable Diffusion 1.5 + optional VAE/LoRA
├── Models/                # Place your local .safetensors files here
│   ├── ponyDiffusionV6XL_v6StartWithThisOne.safetensors
│   ├── sdxl_vae.safetensors
│   └── HandFixer_pdxl_Incrs_v1.safetensors
└── Outputs/
    ├── Flux/
    ├── Pony/
    └── SD15/
```

---

## ⚙️ Requirements

- Python **3.10+**
- NVIDIA GPU with CUDA support
  - Minimum **8 GB VRAM** (FLUX uses sequential CPU offload to fit)
  - **12–16 GB** recommended for Pony XL / SDXL
- CUDA Toolkit **11.8** or **12.x**

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Scripts

### 1. `flux.py` — FLUX.1-schnell

Generates images using [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) from Black Forest Labs. The model is downloaded automatically from Hugging Face on the first run.

**Key features:**
- Uses `bfloat16` precision to reduce VRAM usage (~40% less than float32)
- `enable_sequential_cpu_offload()` keeps peak VRAM under 8 GB
- Only 1–4 inference steps needed (very fast)
- No negative prompt support (native FLUX behaviour)

**Configuration** (edit at the top of the file):

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `FLUX.1-schnell` | HuggingFace model ID |
| `PROMPT` | *(example)* | Generation prompt |
| `WIDTH` / `HEIGHT` | `832 × 1216` | Output resolution |
| `STEPS` | `4` | Inference steps (1–4 for schnell) |
| `GUIDANCE` | `0.0` | CFG scale (ignored by schnell) |
| `SEED` | `None` | Fixed seed or `None` for random |
| `OUTPUT` | `Outputs/Flux/output.png` | Output file path |

**Run:**
```bash
python ImageGenerators/flux.py
```

> **Tip:** Swap `MODEL_ID` to `"black-forest-labs/FLUX.1-dev"` for higher quality (requires HuggingFace access approval). Set `STEPS` to 20–50 and `GUIDANCE` to 3.5+.

---

### 2. `pony.py` — Pony Diffusion XL

Generates images using a locally stored **Pony Diffusion V6 XL** checkpoint (`.safetensors`), with a custom **SDXL VAE** and the **HandFixer LoRA** to improve hand anatomy.

**Key features:**
- Loads model from a local `.safetensors` file (no internet required after first setup)
- Custom VAE for improved colour/sharpness
- LoRA fusion at configurable scale
- Uses `EulerAncestralDiscreteScheduler`

**Configuration** (edit at the top of the file):

| Variable | Default | Description |
|---|---|---|
| `model_path` | `Models/ponyDiffusion...` | Local base model path |
| `vae_path` | `Models/sdxl_vae.safetensors` | Local VAE path |
| `lora_path` | `Models/HandFixer...` | Local LoRA path |
| `output_path` | `Outputs/Pony/output.png` | Output file path |
| `STEPS` | `30` | Inference steps |
| `GUIDANCE` | `7.0` | CFG scale |
| `LORA_SCALE` | `0.75` | LoRA fusion strength |
| `SEED` | `None` | Fixed seed or `None` for random |

**Run:**
```bash
python ImageGenerators/pony.py
```

> **Tip:** Pony Diffusion uses score-based quality tags in the prompt (`score_9`, `score_8_up`, etc.). Include them for best results.

---

### 3. `sd15.py` — Stable Diffusion 1.5

Generates images using **Stable Diffusion 1.5** loaded directly from Hugging Face, with an optional custom VAE and an optional LoRA.

**Key features:**
- Model downloaded and cached automatically from HuggingFace
- Optional VAE override (`stabilityai/sd-vae-ft-mse` by default)
- Optional local LoRA loading (set `lora_path = None` to skip)
- Uses `EulerAncestralDiscreteScheduler`

**Configuration** (edit at the top of the file):

| Variable | Default | Description |
|---|---|---|
| `model_id` | `runwayml/stable-diffusion-v1-5` | HuggingFace model ID |
| `vae_id` | `stabilityai/sd-vae-ft-mse` | HuggingFace VAE ID (or `None`) |
| `lora_path` | `None` | Local LoRA `.safetensors` (or `None`) |
| `output_path` | `Outputs/SD15/output.png` | Output file path |
| `WIDTH` / `HEIGHT` | `512 × 512` | Output resolution |
| `STEPS` | `25` | Inference steps |
| `GUIDANCE` | `7.0` | CFG scale |
| `LORA_SCALE` | `0.8` | LoRA fusion strength |
| `SEED` | `None` | Fixed seed or `None` for random |

**Run:**
```bash
python ImageGenerators/sd15.py
```

---

## 🧠 VRAM Reference

| Script | Minimum VRAM | Notes |
|---|---|---|
| `flux.py` | ~8 GB | Sequential CPU offload enabled |
| `pony.py` | ~10–12 GB | SDXL-based |
| `sd15.py` | ~4–6 GB | SD 1.5-based |

---

## 🔧 Troubleshooting

**Out of Memory (OOM) on FLUX:**
Uncomment one or both of these lines in `flux.py`:
```python
pipe.enable_attention_slicing()
pipe.vae.enable_tiling()
```

**Model not found (Pony):**
Make sure the `.safetensors` files are placed in the `Models/` folder and the paths in the script match exactly.

**CUDA not available:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If this prints `False`, verify your CUDA driver and PyTorch installation.

---

## 📄 License

Scripts provided as-is for personal/research use. Refer to each model's own license for usage restrictions:
- [FLUX.1-schnell License](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
- [Stable Diffusion 1.5 License](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- Pony Diffusion XL: check the model card on CivitAI
