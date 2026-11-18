# 独立的 ragen2 conda 环境

`ragen2`是独立conda环境，不继承factory的site-packages，也不使用原RAGEN2 `.venv`。`ragen` editable指向当前项目，`verl` editable指向项目内固定的0.6.1后端（commit `d62da4950573d7a4b7ef2362337952e7ab59e78d`）。

## 日常使用

```bash
conda activate ragen2
cd /path/to/RAGEN2
python -c 'import sys,ragen,verl; print(sys.executable); print(ragen.__file__); print(verl.__file__)'
bash run_ragen2.sh scripts/check_ragen2_environment.py
```

三个路径应分别属于ragen2解释器、当前项目的`ragen/`和当前项目固定的`verl/verl/`。检查脚本默认读取配置中的本地Qwen3-4B tokenizer；模型在其他位置时，传`--model-path /path/to/Qwen3-4B`。它不生成模型回答，不启动Ray或GPU上下文，也不读取API密钥。

`run_ragen2.sh`固定选择ragen2，清除继承的Python路径并重新指定本项目；未激活时使用`conda run -n ragen2`，不会回退到gift1、factory或旧`.venv`。`run_factory.sh`现在只是兼容名称，同样转到`run_ragen2.sh`，因此现有训练、merge及smoke脚本仍能使用原入口。

安装器仅为新环境设置以下四个变量，激活时生效，退出时由Conda恢复先前值：

| 变量 | 配置 | 作用 |
|---|---|---|
| `PYTHONPATH` | `<project>/verl:<project>` | 固定当前项目及固定verl的来源 |
| `PYTHONNOUSERSITE` | `1` | 禁用用户site-packages |
| `PYTHONSAFEPATH` | `1` | 防止当前工作目录自动插入而覆盖已登记源码 |
| `RAGEN_API_KEYS_PATH` | `<project>/ragen/env/api-keys.json` | 指定凭据文件路径，不在环境变量中放密钥内容 |

路径在安装时由项目目录推导，分享的`environment.yml`没有本机prefix或绝对源码URL。项目移动后须重新登记editable与这四项本机路径；不要把带绝对路径的原始Conda导出直接作为通用GitHub配置。

## 沿用math、nash、nmt配置

**激活ragen2只切换Python和包来源，不会自动改旧脚本里的模型、GPU或输出路径。** 原`math.sh`、`nash.sh`等仍含旧模型/卡号/目录，不能直接把它们当作Qwen3启动器。使用新包装器读取同一旧实验YAML和原脚本实际使用的GRPO开关：

```bash
# 只解析最终配置；不会训练、分配GPU或创建输出目录。
bash scripts/train_qwen3_legacy.sh math --gpus 4 --num-gpus 1 --cfg job --resolve
bash scripts/train_qwen3_legacy.sh nash --gpus 4 --num-gpus 1 --cfg job --resolve
bash scripts/train_qwen3_legacy.sh nmt  --gpus 4 --num-gpus 1 --cfg job --resolve
```

对应配置依次是 `_10_math`、`_16_nash_new`、`_29_nmt_batch`。任务配置继续提供环境、批量、长度和保存频率。包装器额外应用原脚本的 GRPO 设置：`adv_estimator=grpo`、`reward_normalization.method=mean_std`、`use_kl_loss=True`。

模型设置集中在 `config/model_profile/qwen3_4b_nothink.yaml`。`run_qwen3_rl.sh` 自动选用它；自定义 Hydra 启动脚本也可以传入一项 `model_profile=qwen3_4b_nothink`，与自己的 GRPO 参数一起使用。通用任务配置未选 profile 时仍保持原来的模型、格式和采样默认值。

该 profile 使用 Qwen3-4B 原生非思考模板，允许在唯一的末尾 `<answer>` 块前解释。parser 只向环境传答案，batch 复用实际采样的正文 token。训练和验证都按 `temperature=0.7`、`top_p=0.8`、`top_k=20` 采样，当前 vLLM 的 `min_p` 默认是 0。训练和验证的采样字段可分别用 Hydra CLI 覆盖。profile 不设置 GPU、输出目录、训练步数或 GRPO。

真正启动前先完成资源核验/交接，指定本轮GPU数量和一个尚不存在的新输出目录；脚本不会自动找空闲卡，GPU列表长度必须与数量相同：

```bash
# GPU_IDS、NUM_GPUS、NEW_OUTPUT_DIR需先按本轮资源和实验预算设置。
bash scripts/train_qwen3_legacy.sh math \
  --gpus "$GPU_IDS" --num-gpus "$NUM_GPUS" --output-dir "$NEW_OUTPUT_DIR" \
  trainer.total_training_steps=200
```

`--output-dir` 防止误续跑或覆盖旧实验。上述命令展示接口，不代表原批量和长度已在任意单卡通过显存验收。要选用兼容的同系列 HF checkpoint，可设置 `RAGEN_MODEL_PATH`。Math 和 Nash 须使用各自的输出目录。

## 安装与可分享配置

- `environment.yml`：Python 3.12.13、pip 25.0.1的独立Conda创建入口。
- `requirements-ragen2.txt`：Linux x86_64 / CPython3.12 / CUDA12.8的直接依赖版本。核心为Torch2.8.0、vLLM0.11.0、Transformers4.56.1、FlashAttention2.8.1、NumPy1.26.4、SciPy1.14.1。
- `requirements.txt`转到上述固定版本入口；`requirements-ragen2-linux-cu128.snapshot.txt`保存本轮389个已安装发行包中除本地ragen/verl之外的版本快照，不含服务器绝对editable路径。它记录已安装状态，不是跨平台依赖求解锁文件。
- `scripts/setup_ragen2_env.sh`：只新建或显式配置固定名ragen2，核对后端源码并登记两个editable包；默认拒绝覆盖已有环境。

在同一服务器复用已验证二进制时：

```bash
bash scripts/setup_ragen2_env.sh --clone-source factory
# 仅当本轮已先创建独立ragen2副本时：
bash scripts/setup_ragen2_env.sh --configure-existing
```

创建使用`--copy`，随后只修改新prefix。默认允许下载Conda缓存中缺少的包；缓存完整时才使用可选的`--offline-clone`。安装与修复均已在新环境实测，包括：

1. 清理复制来的外部editable metadata、`__editable__*`、`_editable_impl_*`、egg-link和外部`.pth`路径，不删除外部源码目录。
2. 移除不属于此训练栈、与`outlines-core`冲突的旧`outlines`，保留vLLM需要的`outlines-core`。
3. 移除要求NumPy2的桌面`opencv-python`，强制重装`opencv-python-headless==4.11.0.86`，恢复两个OpenCV发行包共享的`cv2`文件并兼容NumPy1.26.4。
4. 对克隆导致的重复`packaging`/`setuptools`记录，清理目标prefix内旧metadata，并重装固定版本的实际包代码；仅删metadata不能修复代码版本。

factory、gift1、EnvFactory及原RAGEN没有被这些操作改动。新环境内的修复记录和最终依赖快照由部署证据保留。

新机器从配置创建时，先取得项目及固定verl，再准备匹配CPython3.12、Torch2.8、CUDA12和C++ ABI的FlashAttention2.8.1本地wheel：

```bash
bash scripts/setup_ragen2_env.sh \
  --flash-attn-wheel /path/to/matched/flash_attn-2.8.1-....whl
```

此路径从`environment.yml`创建Python，再用pip安装CUDA12.8二进制栈。`--requirements FILE`可使用已审阅的同平台完整快照。当前不把克隆路径的验收等同于新机器fresh重建成功；驱动、ABI和目标训练需另行验证。`pip freeze`只是已安装清单，不是跨平台求解锁文件。API密钥不进入这些文件、安装输出或GitHub。

## 本轮验收状态

新conda环境已通过独立来源/路径检查、`pip check`、18项旧数学与梯度逐元素对照、9项原生模板CPU检查、math/nash/nmt三个最终配置解析，以及本地Math环境CPU检查。激活后从`/tmp`检查来源，解释器、二进制包与两个editable源码均符合预期；训练主进程和实际Ray worker的`/proc/<pid>/exe`均属于新ragen2 prefix。

本轮在GPU4（A800-SXM4-80GB）用Qwen3-4B原生非思考完成两步**单卡全参数GRPO**，退出码0。沿用`qwen3_4b_nothink_smoke`的小配置：2组×4轨迹、微批1、上下文2048、生成128、actor/optimizer/ref CPU offload、gradient checkpoint、vLLM预算0.35。首步PG loss为-0.0104871914，8条归一化奖励均非零；抽检37,888个参数中首步37,290、次步37,478个变化，最大绝对delta约2.03e-6和2.00e-6。次步同组奖励相等、优势为0，不能将其PG loss=0误记成未执行更新。

16条真实轨迹全部通过独立检查：native `enable_thinking=False`前缀、冻结旧parser、raw→canonical、拼接tokens、next-token shift、旧assistant mask、positions与state奖励归一化均一致；两批有效动作分别2/8和8/8，未放宽解析规则。125个上轮已审计训练源文件/配置哈希不变；此次未因换环境修改训练算法或输入输出模板。

step2 checkpoint和CPU HF导出可读，399个张量的分片/索引、scheduler epoch2、chat template和generation配置均通过验收。远端完整证据：`/home/lvnuoyan/RAGEN2/logs/ragen2_conda_20261117/`；短训日志、原始轨迹、checkpoint与HF导出：`/data1/lvnuoyan/ragen2_runs/conda_validation_20261117/`。GPU4已由tmux `ragen2_conda_gpu4_20261117`恢复原gift1的7B/.90/127.0.0.1:18440长期服务，health/models/短推理均200，显存约72073MiB。

这里证明新环境可加载、生成、计算loss并更新Qwen3-4B；不是正式效果实验或所有资源配置的显存承诺。math/nash/nmt本轮仅验收最终配置，本地Math另有CPU数据和gold→step验证，未运行其正式长训练、外部对手或多卡训练；多轮模板仅CPU验收。smoke保存model与scheduler/RNG，不保存optimizer；原有正式保存配置不因此修改。

参考：[Conda环境创建与复制](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)、[vLLM0.11.0 CUDA12.8二进制说明](https://docs.vllm.ai/en/v0.11.0/getting_started/installation/gpu.html)、[pip freeze的边界](https://pip.pypa.io/en/stable/cli/pip_freeze/)。
