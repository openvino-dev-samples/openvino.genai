# FLUX.2-klein on OpenVINO GenAI — Environment Setup & Usage Guide

This document explains how to set up the environment, export a correctly
quantized INT4 FLUX.2-klein model, and run text-to-image / image-editing
inference through the **C++ `Flux2KleinPipeline`** exposed by the
`openvino_genai` Python bindings.

> FLUX.2 [klein] is a step-distilled rectified-flow image model from Black
> Forest Labs. It unifies text-to-image and multi-reference image editing in a
> single 4-step pipeline. The pipeline has four sub-models: a **Qwen3 text
> encoder**, a **Flux2 transformer**, and a **VAE** encoder/decoder.

---

## 1. Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10/11 x64 (Linux works too) |
| RAM | **≥ 32 GB** for export, **≥ 16 GB** for INT4 inference |
| OpenVINO | **2026.3.0** (required by the genai `flux2-klein` branch) |
| Python | 3.12 (matches the prebuilt `py_openvino_genai.cp312` binding) |
| Disk | ~9 GB per exported INT4 model |

The reference environment used here is the editable install at
`D:\optimum-intel\py_env` (the only env with OpenVINO 2026.3.0):

```
D:\optimum-intel\py_env\Scripts\python.exe
```

---

## 2. Python environment for model export

The export uses **optimum-intel** on the `flux.2-klein` branch (which registers
both `OVFlux2KleinPipeline` and the per-component INT4 quantization recipe).

### 2.1 Switch optimum-intel to the `flux.2-klein` branch

```powershell
cd D:\optimum-intel
git checkout flux.2-klein
# verify the pipeline class is importable
D:\optimum-intel\py_env\Scripts\python.exe -c "from optimum.intel import OVFlux2KleinPipeline; print('OK')"
```

### 2.2 Dependency versions (important)

| Package | Version | Why |
|---------|---------|-----|
| `openvino` | `>= 2026.0` (2026.3.0 here) | klein support |
| `diffusers` | `@153fcbc5a81be884304dcbe64986b7ea917a3802` | Flux2Klein algorithm |
| `optimum-intel` | `@flux.2-klein` branch | OV export + quant recipe |
| `transformers` | **`>= 4.51, < 5`** (e.g. 4.57.x) | see note below |
| `nncf` | `>= 2.15.0` | INT4 weight compression |

> ⚠ **transformers 5.x breaks Qwen3 TorchScript tracing** during export
> (`sdpa_mask_without_vmap() missing 1 required positional argument:
> 'cache_position'`). Use a **4.x** release for the export step:
>
> ```powershell
> D:\optimum-intel\py_env\Scripts\python.exe -m pip install "transformers>=4.51,<5"
> ```

---

## 3. Export a correct INT4 model

The quantization recipe is **registered by model id / folder name** in
`optimum/intel/openvino/configuration.py` under
`_DEFAULT_4BIT_WQ_CONFIGS["black-forest-labs/FLUX.2-klein-4B"]`:

| Sub-model | Precision |
|-----------|-----------|
| transformer | INT4 (`sym=False`, `group_size=128`, `ratio=0.8`, `group_size_fallback="ignore"`) |
| text_encoder | INT4 (same as transformer) |
| vae_encoder / vae_decoder | INT8 |

`group_size_fallback="ignore"` keeps the transformer's `pos_embed`/`aten::outer`
ops (channel_size=1, cannot be grouped at 128) in **fp16** instead of forcing
them into INT4 — this is what prevents the over-quantized, near-white output.

> The lookup matches by **folder name**, so exporting from a local directory
> named `FLUX.2-klein-4B` automatically picks up the correct recipe — no
> re-download required.

### 3.1 Export command

```powershell
$env:OPENVINO_TELEMETRY_OPT_OUT = "1"
& "D:\optimum-intel\py_env\Scripts\optimum-cli.exe" export openvino `
    --model "D:\models\FLUX.2-klein-4B" `
    --task text-to-image `
    --weight-format int4 `
    "D:\models\FLUX.2-klein-4B-OV-INT4-nb"
```

A correct export prints a mixed-precision summary like:

```
int4_asym, group size 128 | 80% (...)
int8_asym, per-channel    | 20% (...)
float                     |  0% (a few pos_embed ops)
```

### 3.2 Verify the transformer recipe (sanity check)

```powershell
D:\optimum-intel\py_env\Scripts\python.exe -c "import openvino as ov; wc=ov.Core().read_model(r'D:\models\FLUX.2-klein-4B-OV-INT4-nb\transformer\openvino_model.xml').get_rt_info()['nncf']['weight_compression']; print(wc['mode'].value, 'ratio', wc['ratio'].value, 'group_size', wc['group_size'].value)"
# expected: int4_asym ratio 0.8 group_size 128
```

A **broken** model shows `int8_asym ratio 1.0 group_size -1` and produces
near-white images.

### 3.3 Exported model layout

```
FLUX.2-klein-4B-OV-INT4-nb/
├── model_index.json            # _class_name = "Flux2KleinPipeline"
├── openvino_config.json
├── vae_bn_stats.npz            # BN running_mean / running_var (eps 1e-4)
├── scheduler/
├── text_encoder/               # Qwen3 → prompt_embeds (3 layers x 2560 = 7680)
├── tokenizer/                  # includes openvino_tokenizer/detokenizer .xml/.bin
├── transformer/                # INT4
├── vae_decoder/                # INT8
└── vae_encoder/                # INT8
```

---

## 4. Verify the model with the Python reference (optional)

```powershell
$env:OPENVINO_TELEMETRY_OPT_OUT = "1"
D:\optimum-intel\py_env\Scripts\python.exe D:\openvino.genai\tools\_verify_nb_int4.py `
    "D:\models\FLUX.2-klein-4B-OV-INT4-nb" `
    "D:\openvino.genai\tools\flux2_klein_nb_int4.png"
```

A correct model gives a **real image** (pixel `std ≈ 56`, full 0–244 range).
A broken model gives near-white output (pixel `std ≈ 0.5`).

---

## 5. Build the C++ genai pipeline

The `Flux2KleinPipeline` is built into the core `openvino_genai` library and is
dispatched automatically via `model_index.json` `_class_name`; the existing
`Text2ImagePipeline` / `Image2ImagePipeline` bindings route to it (no new
binding classes needed).

```powershell
cd D:\openvino.genai\build
cmake -DENABLE_PYTHON=ON -DPython3_EXECUTABLE="D:\optimum-intel\py_env\Scripts\python.exe" ..
cmake --build . --config Release --target openvino_genai
cmake --build . --config Release --target py_openvino_genai
```

Output binding:
`D:\openvino.genai\build\openvino_genai\py_openvino_genai.cp312-win_amd64.pyd`

---

## 6. Run inference (C++ pipeline via bindings)

Set `PYTHONPATH` to the build directory so `import openvino_genai` resolves to
the freshly built binding, and use the OpenVINO 2026.3.0 Python:

```powershell
$env:OPENVINO_TELEMETRY_OPT_OUT = "1"
$env:PYTHONPATH = "D:\openvino.genai\build"
```

### 6.1 Text-to-Image

```powershell
D:\optimum-intel\py_env\Scripts\python.exe D:\openvino.genai\tools\flux2_klein_smoke_test.py `
    --model "D:\models\FLUX.2-klein-4B-OV-INT4-nb" `
    --prompt "A cat holding a sign that says hello world" `
    --height 512 --width 512 `
    --steps 4 --guidance-scale 1.0 --seed 0 `
    --device CPU `
    --output-dir "D:\openvino.genai\tools"
# → D:\openvino.genai\tools\flux2_klein_t2i.png
```

### 6.2 Image editing (multi-reference)

```powershell
D:\optimum-intel\py_env\Scripts\python.exe D:\openvino.genai\tools\flux2_klein_smoke_test.py `
    --model "D:\models\FLUX.2-klein-4B-OV-INT4-nb" `
    --edit-image "D:\models\FLUX.2-klein-4B\editing.jpg" `
    --edit-prompt "Make the background a sunny beach" `
    --steps 4 --guidance-scale 1.0 --seed 0 `
    --device CPU `
    --output-dir "D:\openvino.genai\tools"
# → D:\openvino.genai\tools\flux2_klein_edit.png
```

### 6.3 Minimal Python snippet

```python
import openvino_genai as ov_genai
import numpy as np
from PIL import Image

pipe = ov_genai.Text2ImagePipeline(
    r"D:\models\FLUX.2-klein-4B-OV-INT4-nb", "CPU"
)
result = pipe.generate(
    "A cat holding a sign that says hello world",
    width=512, height=512,
    num_inference_steps=4,   # klein is 4-step distilled
    guidance_scale=1.0,      # distilled → no CFG
    rng_seed=0,
)
Image.fromarray(result.data[0].astype(np.uint8)).save("out.png")
```

**Recommended generation parameters (distilled model):**

| Parameter | Value |
|-----------|-------|
| `num_inference_steps` | 4 |
| `guidance_scale` | 1.0 (no classifier-free guidance) |
| `height` / `width` | multiples of 16 (e.g. 512, 1024) |

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Near-white image (pixel std ≈ 0.5) | Transformer over-quantized (`int8_asym ratio 1.0 group_size -1`) | Re-export with the registered recipe (§3); verify `int4_asym ratio 0.8 group_size 128` |
| `sdpa_mask_without_vmap() missing ... cache_position` during export | transformers 5.x incompatible with Qwen3 tracing | `pip install "transformers>=4.51,<5"` |
| `OVFlux2KleinPipeline` import fails | optimum-intel not on `flux.2-klein` branch | `git checkout flux.2-klein` in `D:\optimum-intel` |
| `import openvino_genai` resolves to wrong build | `PYTHONPATH` not set | `$env:PYTHONPATH = "D:\openvino.genai\build"` |
| `scaling_factor attribute is missing from VAE` warning | benign — BN handled outside the VAE graph via `vae_bn_stats.npz` | ignore |
| `Multiple distributions found for package optimum` warning | editable optimum-intel + optimum metadata | benign — ignore |

---

## 8. Notes

- klein is **4-step distilled**; it does not use classifier-free guidance
  (`guidance_scale=1.0`). Higher step counts do not improve a broken model.
- VAE BatchNorm normalization is applied **outside** the VAE graph in the
  pipeline using `vae_bn_stats.npz`
  (`norm = (x - mean) / sqrt(var + 1e-4)`, denorm is the inverse).
- The text encoder output (`prompt_embeds`, dim 7680 = 3 Qwen3 layers × 2560)
  legitimately contains very large activations (Qwen3 massive activations) —
  this is expected, not a bug.
- After export you may restore `transformers` to your previous version if other
  workflows require it.
