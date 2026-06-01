# Z-Image-Turbo INT4 — 从源码编译到推理使用说明

> 适用仓库：https://github.com/openvino-dev-samples/openvino.genai.git（分支 `z-image`）  
> 默认示例模型（ModelScope 预转换 INT4）：https://www.modelscope.cn/models/snake7gun/Z-Image-Turbo-int4-ov  
> 本文不包含 optimum-intel 模型转换步骤。

---

## 一、前置条件

| 组件 | 最低版本 |
|------|---------|
| OS | Windows 10/11 x64 |
| Python | 3.10+ |
| CMake | 3.23+ |
| Visual Studio | 2019（v16.3）或 2022，需勾选"使用 C++ 的桌面开发" |
| Git | 任意近期版本 |

---

## 二、克隆仓库并切换到 z-image 分支

```powershell
git clone https://github.com/openvino-dev-samples/openvino.genai.git
cd openvino.genai
git checkout z-image
git submodule update --init --recursive
```

---

## 三、创建 Python 虚拟环境并安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt
python -m pip install openvino openvino-tokenizers pillow numpy
```

> `requirements-build.txt` 包含 cmake、pybind11-stubgen 等构建工具依赖。
> `openvino` 和 `openvino-tokenizers` 提供运行时，以及 CMake 查找所需的 cmake 配置文件。

---

## 四、编译 OpenVINO GenAI（含 Python 绑定）

### 4.1 配置 CMake

```powershell
# OpenVINO 安装在 .venv 中，从中获取 cmake 配置目录
$ov_cmake = (python -c "import openvino, os; print(os.path.join(os.path.dirname(openvino.__file__), 'cmake'))")

cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
    -DOpenVINO_DIR="$ov_cmake" `
    -DENABLE_PYTHON=ON `
    -DENABLE_TESTS=OFF `
    -DENABLE_SAMPLES=ON
```

> 如果使用 Visual Studio 2019，将 `"Visual Studio 17 2022"` 改为 `"Visual Studio 16 2019"`。

### 4.2 构建 Python 模块

```powershell
cmake --build build --config Release --target py_openvino_genai --parallel
```

构建完成后 `.pyd` 和 `.dll` 位于 `build\openvino_genai\Release\`（或 `build\openvino_genai\`）。

### 4.3 设置运行时环境变量

每次新开终端前执行（或写入 profile）：

```powershell
# Python 能找到 openvino_genai 包
$env:PYTHONPATH = "$PWD\src\python;$env:PYTHONPATH"

# 系统能找到 openvino_genai.dll
$env:PATH = "$PWD\build\openvino_genai\Release;$env:PATH"
```

### 4.4 快速自检

```powershell
python -c "
import openvino_genai as g
print('openvino_genai version:', g.__version__)
print('ZImageTransformer2DModel:', hasattr(g, 'ZImageTransformer2DModel'))
print('ZImageTextEncoder:       ', hasattr(g, 'ZImageTextEncoder'))
print('Text2ImagePipeline:      ', hasattr(g, 'Text2ImagePipeline'))
"
```

正常输出示例：

```
openvino_genai version: 2026.2.0
ZImageTransformer2DModel: True
ZImageTextEncoder:        True
Text2ImagePipeline:       True
```

---

## 五、下载预转换模型（ModelScope INT4）

模型主页：https://www.modelscope.cn/models/snake7gun/Z-Image-Turbo-int4-ov

### 方式 A：ModelScope SDK（推荐）

```powershell
pip install modelscope
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('snake7gun/Z-Image-Turbo-int4-ov', cache_dir=r'D:\models')
"
```

下载完成后模型目录为：

```
D:\models\snake7gun\Z-Image-Turbo-int4-ov\
```

### 方式 B：git-lfs 手动克隆

```powershell
git lfs install
git clone https://www.modelscope.cn/snake7gun/Z-Image-Turbo-int4-ov.git D:\models\Z-Image-Turbo-int4-ov
```

模型目录应包含以下子目录/文件（缺少任何一个推理将失败）：

```
model_index.json
scheduler\scheduler_config.json
text_encoder\openvino_model.xml / .bin
tokenizer\*
transformer\openvino_model.xml / .bin
vae_decoder\openvino_model.xml / .bin
```

---

## 六、推理示例

新建文件 `run_zimage.py`：

```python
import numpy as np
from PIL import Image
import openvino_genai as ov_genai

# 按实际路径修改
MODEL_DIR = r"D:\models\snake7gun\Z-Image-Turbo-int4-ov"
PROMPT     = "A photo of a red panda wearing a tiny straw hat, sitting on a wooden table, soft studio lighting"
DEVICE     = "CPU"          # 改为 "GPU" 可使用 Intel GPU 加速
SEED       = 42
STEPS      = 9
WIDTH      = 1024
HEIGHT     = 1024

pipe = ov_genai.Text2ImagePipeline(MODEL_DIR, DEVICE)

def step_callback(step, num_steps, latent):
    print(f"  step {step + 1}/{num_steps}")
    return False  # 返回 True 可提前终止

image_tensor = pipe.generate(
    PROMPT,
    width=WIDTH,
    height=HEIGHT,
    num_inference_steps=STEPS,
    guidance_scale=0.0,           # Z-Image-Turbo 使用 0.0
    generator=ov_genai.TorchGenerator(SEED),
    callback=step_callback,
)

arr = np.array(image_tensor.data, copy=True).reshape(image_tensor.get_shape())
out_path = "zimage_out.png"
Image.fromarray(arr[0]).save(out_path)
print(f"已保存：{out_path}  shape={arr[0].shape}")
```

运行：

```powershell
python run_zimage.py
```

---

## 七、切换设备

| 设备 | 说明 |
|------|------|
| `"CPU"` | 默认，全平台可用 |
| `"GPU"` | Intel 核显 / Arc 独显，驱动须支持 OpenVINO GPU plugin |
| `"GPU.0"` / `"GPU.1"` | 多 GPU 场景指定具体显卡 |
| `"NPU"` | Intel NPU（部分平台） |

修改示例中 `DEVICE = "GPU"` 即可，其余代码无需改动。

---

## 八、常见问题

### Q1：`ModuleNotFoundError: No module named 'openvino_genai'`

- 确认已激活虚拟环境 `.venv`
- 确认已设置 `$env:PYTHONPATH`（见 §4.3）
- 确认编译目标为 `py_openvino_genai`（见 §4.2）

### Q2：运行时报"找不到 DLL"

- 确认已设置 `$env:PATH` 指向 `build\openvino_genai\Release`（见 §4.3）

### Q3：首次推理异常缓慢

- 首次运行需编译 OpenVINO IR，属正常现象。GPU 设备首次编译尤其耗时，后续运行会快速许多。

### Q4：输出图片全黑或全是噪点

- 检查模型目录是否完整（尤其 `model_index.json`、`transformer`、`text_encoder`、`vae_decoder`）。
- 确认使用 `guidance_scale=0.0`。
- 确认 `num_inference_steps=9`（Z-Image-Turbo 为 step-distilled 模型，步数不宜过少也不宜过多）。

### Q5：CMake 报 `Could not find OpenVINO`

```powershell
# 打印正确的 cmake 目录
python -c "import openvino, os; print(os.path.join(os.path.dirname(openvino.__file__), 'cmake'))"
```

将输出路径直接传给 `-DOpenVINO_DIR`。
