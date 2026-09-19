# SenseNova-U1.5 FP16 / BF16 与 FP8 离线评测

流程是固定输入与参数，依次通过 `with OmniRunner(...)` 运行未量化基线和 online FP8，再输出指标表、保存 `metrics.json`，最后检查输出有效性及外部提供的阈值。没有自写进程管理或独立 CLI，也不需要 HTTP server。

**当前 FP16 不是有效的图像 baseline。** 在本次 A800 软件栈上，真实 FP16 基线和基础 dtype 为 `float16` 的 online FP8 生成结果均为黑图。代表性苹果样本的数值追踪已定位：两组均在第 16 个去噪 step、GEN block 41 的 MLP 残差相加首次出现 Inf，随后 RMSNorm 产生 NaN，并扩散至最终图像；转为 uint8 时才出现 `invalid value encountered in cast`。实际操作数的 FP32 求和与归一化回放保持有限，但尚未实施或验证完整模型修复。两侧理解题均为 5/5，仅说明这 5 题 smoke 的结果，不能证明图像路径正确。代码仍默认选择 FP16（`float16`），BF16（`bfloat16`）可选且已有正常图像对照。

在仓库根目录运行；下面显式选择已有正常图像对照的 BF16：

```bash
SENSENOVA_RUN_FP8_EVAL=1 \
SENSENOVA_BASELINE_DTYPE=bf16 \
SENSENOVA_U15_MODEL=/path/to/SenseNova-U1.5-8B-MoT \
SENSENOVA_FP8_OUTPUT_DIR=/path/to/new-empty-output \
pytest tests/e2e/accuracy/test_sensenova_u1.py -s
```

省略 `SENSENOVA_BASELINE_DTYPE` 或设为 `fp16` 即选择 FP16；默认不启用评测。输出目录必须不存在或为空，省略时自动创建带时间戳的目录。未安装 pytest-xdist 的环境可追加 `-o addopts=''`。

已有 A800 软件栈需要在同一命令前增加 `VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel`，两侧保持相同环境。这只针对该栈错误选择 CUTLASS FP8 路径的情况，不是其他 GPU 的默认要求。

两侧固定 checkpoint、prompt、seed 和所选 dtype，仅 FP8 组增加 `quantization=fp8`。固定 TP=1、batch=1、`TORCH_SDPA`、eager、paged decode 关闭，无 LoRA、diffusion cache backend 或 offload；PyTorch SDPA 子 kernel 自动选择。出图参数为 `think=false`、CFG=4、`cfg_norm=none`、`cfg_interval=[0,1]`、`timestep_shift=3`、`t_eps=0.02`，均保存到 `config.json`。

样本与指标：

- **生成：** 默认苹果（seed 42）、三只鸭子（seed 42）、蝴蝶（seed 123），共 3 组 1024×1024、50-step 对照；使用原分辨率 RGB SSIM、PSNR。
- **可选 LPIPS：** `SENSENOVA_FP8_LPIPS=1` 启用 AlexNet LPIPS@256，需要额外依赖和权重，首次使用可能下载。图像指标只用于生成图片，不用于理解文本。
- **理解：** 两张固定 512×512 合成图（红正方形、两个蓝圆）的 4 道 I2T 题，加 `2+2` 的 1 道 T2T 题。固定 greedy、`max_tokens=768`、`temperature=0`，提取最终答案并做 normalized exact match；这不是 VQA benchmark。
- **自定义生成集合：** `SENSENOVA_FP8_CASES=/path/to/cases.json` 接受非空 JSON 数组，每项含唯一 `id`、`prompt`、整数 `seed`，可复用此前 30-case 文件。理解题仍为固定 5 题。

默认非纯色用例若输出纯色图，则标为 `invalid`，不计算该对图像的相似度，避免两张黑图得到虚假的高分。A800 的 FP16 对照已验证这一路径：6 张黑图全部被拦截，保留理解结果后 pytest 失败。只有用例本来就要求纯色时，才在该用例中设置 `allow_uniform=true`。空最终答案或未闭合思考块也标为无效输出。

终端分别打印生成指标表、理解答案表及精确匹配率；保存 `config.json`、合成输入、各组生成图片和 `metrics.json`。基线目录（`fp16/` 或 `bf16/`）与 `fp8/` 下分别保存 `answers.json`，逐题保留理解文本与答案。先保留指标与无效原因，再执行断言；未配置阈值时，运行成功不代表量化质量验收通过。

`SENSENOVA_FP8_THRESHOLDS=/path/to/thresholds.json` 可提供 `ssim_min`、`psnr_min_db`、`lpips_max`、`understanding_min_accuracy`（0–1）。没有任何默认数值；`lpips_max` 要求启用 LPIPS。未提供阈值且输出有效时只做 baseline measurement。当前直接使用 OmniRunner 的 BF16／FP8 入口已在 A800 完成验证：3 组图像指标有效，理解题两侧各 5/5，`gate.status=not_configured`。这不覆盖 FP16 图像路径，也不代表质量验收通过；原厂精度对齐尚未验证，入口尚未接入 CI 调度。
