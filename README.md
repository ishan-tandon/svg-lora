# SVG-LoRA: LoRA Fine-Tuning Qwen for Text-to-SVG Generation

Fine-tune [Qwen 2.5 Instruct](https://huggingface.co/Qwen) models using LoRA to generate SVG code from natural language descriptions. Built for the NYU SVG Contest on Kaggle.

This repo contains two notebook stubs — one for training from scratch and one for continuing training from an existing adapter. Both are designed to run as-is on **Google Colab with an A100 GPU**.

---

## Repository Structure

```
├── train notebook base.ipynb        # Train a LoRA adapter from a base Qwen model
├── train notebook continue.ipynb    # Continue training from an existing LoRA adapter
├── inference.py                     # Standalone inference script
├── data_clean.ipynb                 # Data cleaning and preprocessing
├── baseline.ipynb                   # Baseline experiments
├── csv/                             # Data directory
│   ├── train.csv                    # Full training data (not in repo — too large)
│   ├── train_clean.csv              # Cleaned training data (not in repo — too large)
│   ├── sample_submission.csv        # Kaggle submission format
│   └── test.csv                     # Test prompts with 'id' and 'prompt' columns
├── qwen-svg-lora/                   # Checkpoint directory (weights not in repo)
├── qwen-svg-lora-v2/                # Checkpoint directory (weights not in repo)
├── qwen2b_svg_lora_v3/              # Checkpoint directory (weights not in repo)
└── qwen2b_svg_merged/               # Merged model directory (weights not in repo)
```

---

## Quick Start

### 1. Open in Colab

Upload either notebook to Google Colab. Set the runtime to **A100 GPU** with **High RAM**:

> Runtime → Change runtime type → A100 GPU, High RAM

### 2. Install Dependencies

The first cell handles installation automatically via `uv`:

```
unsloth, torch>=2.5, triton>=3.1, bitsandbytes, xformers,
trl, transformers, datasets, pandas, lxml
```

### 3. Upload Your Data

Your training CSV needs two columns: `prompt` (text description) and `svg` (SVG markup). The test CSV needs `id` and `prompt` columns.

Either upload directly to Colab or mount Google Drive and set the paths in CONFIG.

### 4. Run All Cells

The notebook will train, run inference on the test set, and output a `submission.csv`.

---

## Choosing a Notebook

| Notebook | Use When |
|---|---|
| **train notebook base.ipynb** | Training a fresh LoRA adapter on a base Qwen model from scratch |
| **train notebook continue.ipynb** | You already have a trained adapter and want to continue training with more data, cleaned data, or different hyperparameters |

---

## Configuration Reference

Both notebooks use a `CONFIG` dictionary at the top. Here's every parameter and how to set it:

### Paths

| Parameter | Description | Example |
|---|---|---|
| `train_csv` | Path to your training CSV with `prompt` and `svg` columns | `"/content/drive/MyDrive/train_clean.csv"` |
| `test_prompts_csv` | Path to test CSV with `id` and `prompt` columns | `"/content/drive/MyDrive/test.csv"` |
| `output_dir` | Where checkpoints and the final adapter are saved | `"/content/qwen2b_svg_lora"` |
| `submission_path` | Where the submission CSV is written | `"/content/submission.csv"` |
| `v1_adapter_path` | **(continue notebook only)** Path to an existing LoRA adapter to resume from | `"/content/drive/MyDrive/qwen2b_svg_lora"` |

### Model Selection

| Parameter | Description | Default | Notes |
|---|---|---|---|
| `model_name` | HuggingFace model ID or local path | `"unsloth/Qwen3.5-2B"` | Use any Qwen model supported by Unsloth. Larger models (7B, 14B) need more VRAM. The `unsloth/` prefix loads the Unsloth-optimized version. |
| `max_seq_length` | Maximum token sequence length for training | `1024` (base) / `2048` (continue) | Controls how much of each sample the model sees. Longer SVGs need higher values. Increasing this uses more VRAM. |
| `load_in_4bit` | Load model weights in 4-bit quantization | `True` | Cuts VRAM usage roughly in half. Set `False` if you have enough VRAM and want slightly better quality. |

**Choosing a model:** The notebooks default to `Qwen3.5-2B` which fits comfortably on an A100. You can substitute any Qwen model — for example `"unsloth/Qwen2.5-7B-Instruct"` — as long as it fits in memory. The `unsloth/` prefix is recommended for faster training.

### LoRA Hyperparameters

| Parameter | Description | Default | Guidance |
|---|---|---|---|
| `lora_r` | LoRA rank — controls adapter capacity | `32` | Higher = more expressive but slower. Common values: 8, 16, 32, 64. 32 works well for SVG generation. |
| `lora_alpha` | LoRA scaling factor | `64` | Usually set to 2× `lora_r`. Higher values make the adapter's influence stronger. |
| `lora_dropout` | Dropout on LoRA layers | `0` | Set to 0.05–0.1 if you see overfitting. 0 is fine for large datasets. |

**Target modules** are hardcoded to all attention and MLP projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. This gives the adapter maximum expressiveness.

> **Important:** When using the continue notebook, the LoRA settings (`lora_r`, `lora_alpha`, `lora_dropout`) must match the adapter you're loading from. Mismatched settings will cause errors.

### Training Hyperparameters

| Parameter | Description | Base Default | Continue Default | Guidance |
|---|---|---|---|---|
| `max_steps` | Total training steps | `500` | `1500` | More steps = more learning. Watch training loss — stop when it plateaus. Set to `0` for inference-only (see below). |
| `per_device_train_batch_size` | Samples per GPU per step | `32` | `16` | A100 80GB can handle 32 at seq_length 1024. Reduce if you get OOM errors. |
| `gradient_accumulation_steps` | Accumulate gradients over N steps | `2` | `1` | Effective batch size = `batch_size × accumulation_steps`. Higher gives smoother training but slower iteration. |
| `learning_rate` | Peak learning rate | `2e-4` | `5e-5` | **Critical for continued training.** Use a lower LR (2–5×) when continuing to avoid catastrophic forgetting. |
| `warmup_ratio` / `warmup_steps` | Steps before LR reaches peak | `0.05` (ratio) | `50` (steps) | Warms up the learning rate linearly. Prevents early instability. |
| `weight_decay` | L2 regularization | `0.01` | `0.01` | Helps prevent overfitting. 0.01 is standard. |
| `logging_steps` | Log metrics every N steps | `10` | `10` | Lower = more granular loss tracking. |
| `save_steps` | Save a checkpoint every N steps | `100` | `500` | Checkpoints are saved to `output_dir`. Only the 2 most recent are kept (`save_total_limit=2`). |
| `eval_size` | Fraction of data held out for evaluation | `0.01` | `0.01` | 1% is typically enough to monitor overfitting. |

**Other training settings** (hardcoded in the notebooks):
- **Optimizer:** `paged_adamw_8bit` — memory-efficient AdamW
- **LR scheduler:** `cosine` — decays the learning rate smoothly
- **Precision:** Automatically uses `bf16` if supported, otherwise `fp16`
- **Packing:** `False` — packing is disabled because it causes extreme slowness with this setup
- **Gradient checkpointing:** Enabled via Unsloth for memory savings

### Inference Hyperparameters

| Parameter | Description | Base Default | Continue Default | Guidance |
|---|---|---|---|---|
| `max_new_tokens` | Maximum tokens the model can generate per SVG | `4096` | `2048` | Longer SVGs need more tokens. 2048 is usually sufficient. |
| `max_svg_chars` | Hard character limit for output SVGs | `8000` | `8000` | SVGs exceeding this are replaced with a fallback. Matches the contest limit. |
| `temperature` | Sampling randomness | `0.6` | `0.3` | Lower = more deterministic/conservative. Higher = more creative/varied. 0.3–0.6 is the sweet spot for SVG. |
| `top_p` | Nucleus sampling threshold | `0.9` | `0.85` | Lower = more focused sampling. Works in tandem with temperature. |
| `repetition_penalty` | Penalizes repeated tokens | `1.05` | `1.1` | Prevents the model from generating repetitive paths. 1.05–1.15 is a good range. |
| `inference_batch_size` | **(continue notebook only)** Prompts processed in parallel | `8` | `8` | Higher = faster inference but more VRAM. Reduce if OOM. |

---

## Inference-Only Mode (No Training)

To use a pre-trained adapter for inference without any training, set:

```python
"max_steps": 0,
```

Then skip the training cells (Section 6 in base, Section 5 in continue) and jump directly to the inference and submission sections. The model will load with the existing adapter weights and generate SVGs immediately.

For the **base notebook**, you would change `model_name` to point to your adapter path instead of a base model:

```python
"model_name": "/content/drive/MyDrive/qwen2b_svg_lora",  # path to your trained adapter
```

For the **continue notebook**, set `v1_adapter_path` to your adapter and `max_steps` to `0`.

---

## Data Format

### Training CSV

| Column | Type | Description |
|---|---|---|
| `prompt` | string | Natural language description (e.g., "a red circle with a blue border") |
| `svg` | string | Valid SVG markup starting with `<svg` and ending with `</svg>` |

The notebooks automatically clean and filter the data:
- Drops rows with missing prompts or SVGs
- Removes SVGs that don't start with `<svg`
- Filters by character length (configurable via `max_svg_chars`)
- The continue notebook additionally normalizes all SVGs to a 256×256 canvas and filters to 100–4000 characters

### Test CSV

| Column | Type | Description |
|---|---|---|
| `id` | int/string | Unique identifier for each test sample |
| `prompt` | string | Natural language description to generate SVG for |

---

## Chat Template

The model is trained using the ChatML format with a system prompt that constrains SVG output:

```
<|im_start|>system
You are an SVG generation assistant. Given a text description, generate valid SVG code.
Output ONLY the SVG markup. Always use:
<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">.
Use only allowed elements: svg, g, path, rect, circle, ellipse, line, polyline, polygon,
defs, use, symbol, clipPath, mask, linearGradient, radialGradient, stop, text, tspan,
title, desc, style, pattern, marker, filter.
Keep the SVG under 8000 characters.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
{svg}<|im_end|>
```

---

## SVG Validation & Fallbacks

Generated SVGs go through a multi-step pipeline:

1. **Extraction** — regex extracts the first `<svg>...</svg>` block from model output
2. **Truncation repair** — if the SVG is cut off (missing `</svg>`), the notebook attempts to close open tags and complete it
3. **Canvas normalization** — forces `width="256" height="256" viewBox="0 0 256 256"` and adds `xmlns` if missing
4. **Tag validation** — checks that only allowed SVG elements are present (no `<script>`, `<foreignObject>`, etc.)
5. **Fallback** — if all else fails, generates a simple keyword-aware SVG (e.g., "red circle" → red circle shape, "star" → star polygon)

**Allowed SVG elements:** `svg`, `g`, `path`, `rect`, `circle`, `ellipse`, `line`, `polyline`, `polygon`, `defs`, `use`, `symbol`, `clipPath`, `mask`, `linearGradient`, `radialGradient`, `stop`, `text`, `tspan`, `title`, `desc`, `style`, `pattern`, `marker`, `filter`

---

## Saving & Exporting Models

Both notebooks support three export options:

1. **LoRA adapter** (default) — saves only the small adapter weights (~50–200 MB). Load later with `FastLanguageModel.from_pretrained(adapter_path)`.

2. **Merged model** — merges LoRA weights into the base model and saves as a full 16-bit model. Useful for Kaggle offline inference or deployment without Unsloth.

3. **HuggingFace Hub** — push the adapter directly to HuggingFace. Set `PUSH_TO_HF = True` and fill in your username and token.

---

## Known Gotchas

These are hard-won lessons from debugging — documented in the continue notebook:

- **Use `tokenizer.tokenizer`** for inference, not `tokenizer` directly. Unsloth wraps the tokenizer in a `Qwen3VLProcessor` and calling it directly triggers the image processor, causing errors.
- **Set `eos_token_id`** to the `<|im_end|>` token IDs. Without this, the model generates until `max_new_tokens` every time.
- **Decode only new tokens** via `output_ids[0][input_ids.shape[1]:]`. Otherwise you'll get the prompt echoed back.
- **Keep `packing=False`**. Packing causes extreme training slowness with this setup.
- **Use `text_tokenizer.encode()`** not `tokenizer()` to avoid image processor errors during inference.
- **Set `padding_side="left"`** for batched inference.

---

## Hardware Requirements

| Setup | VRAM Needed | Notes |
|---|---|---|
| Qwen 2.5B, 4-bit, batch 32, seq 1024 | ~20 GB | Fits on A100 40GB comfortably |
| Qwen 2.5B, 4-bit, batch 16, seq 2048 | ~30 GB | Recommended for continued training |
| Qwen 7B, 4-bit, batch 8, seq 2048 | ~40 GB | Needs A100 80GB |

For local training, you need an NVIDIA GPU with sufficient VRAM. The notebooks auto-detect BF16 support and fall back to FP16.

---

## License

This project was built for the NYU SVG Generation Contest. Feel free to use and adapt these notebooks for your own experiments.
