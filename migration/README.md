# RAGEN2：Qwen3-4B 非思考兼容迁移

> **2026-11-17 环境更新：** 已改为独立的`ragen2` conda环境；当前安装与启动方法见 [../docs/ragen2_environment.md](../docs/ragen2_environment.md)。`run_factory.sh`仅为`run_ragen2.sh`的兼容别名。下文`.venv`和factory复用描述是首次迁移的历史记录。本轮新环境两步训练与CPU检查的证据位于服务器`logs/ragen2_conda_20261117/`和`/data1/lvnuoyan/ragen2_runs/conda_validation_20261117/`；训练代码、模板与算法未因环境切换改变。

服务器目录：`/home/lvnuoyan/RAGEN2`。基线为服务器 `/home/lvnuoyan/RAGEN` 实际文件，包含未提交改动；132个归档源文件在迁移后再次与服务器比对，一致。原RAGEN训练文件与共享factory环境没有被本轮修改。

## 后端选择

官方RAGEN已更新到verl 0.6.1，但官方代码未显式适配Qwen3的enable_thinking=False和整轨迹模板重渲染。官方main参考commit为9d021fc782a4a9e8e047f13c1cd706b32f16d8f2，其verl固定为d62da4950573d7a4b7ef2362337952e7ab59e78d。

EnvFactory现有源码是verl 0.8.0.dev0，已移除旧DataProto worker与同步rollout接口。直接接该版本需要训练后端重构。本轮采用官方已验证的0.6.1接口路线，复用factory的torch 2.8.0、transformers 4.56.1、vLLM 0.11.0；这是新版verl迁移，不是声称已接入EnvFactory的0.8引擎。

`RAGEN2/verl`指向隔离的固定源码，`.venv`通过system-site-packages复用factory并仅补游戏依赖，以及按verl 0.6.1声明在隔离venv固定numpy 1.26.4及配套SciPy 1.14.1（factory现有SciPy 1.18要求numpy2，不能直接与numpy1混用）。`run_factory.sh`固定源码路径、V1引擎和离线模型读取；不要运行旧安装脚本去重建/降级共享环境。

## 输入输出与算法不变量

- 本轮用户明确确认非思考。Qwen3必须同时设置`data.apply_chat_template_kwargs.enable_thinking=False`和`agent_proxy.enable_think=False`。
- 使用模型实际原生模板推导非思考前缀，每轮保留原生空think块。生成和更新共用同一稳定渲染路径，避免原生整轨迹重渲染丢弃较早前缀。正文与动作仍要求`<answer>…</answer>`。按本轮用户追加确认，仅Qwen3非思考模式把`Let's think step by step and always output:`改为`Always output:`；Qwen2.5提示全文不变，旧解析规则不变。
- Qwen2.5完整使用旧路径。奖励归一化、格式惩罚、整组过滤、singleton GRPO、完整轨迹next-token offset和原assistant mask保持不变。
- 保留旧raw→canonical llm_response规范化后训练的行为；原始输出单独记录。原assistant mask包含角色头、Qwen3强制空think等模板token。本轮不将其改成仅采样completion监督，也不宣称raw tokens逐token on-policy等价。
- 冻结旧数学实现，避免新版GRPO把singleton优势归零、PPO新增logratio clamp、loss聚合默认变化。actor/critic的计算类保留，PCGrad/CST逻辑仍在；这些附加方法本轮未做GPU验收。

## 运行与证据

CPU检查位于`logs/cpu_preflight/`：真实tokenizer单/双轮条件对齐、原生多轮失配负例、Qwen2.5完整packing/mask/reward回归；18项旧数学与梯度逐元素一致；实际训练入口/架构与严格后端配置构造通过。

两步本地Bandit短训练配置为`config/qwen3_4b_nothink_smoke.yaml`，用于框架兼容检查，不是正式游戏实验。GPU结果、原始轨迹、更新证据和占卡恢复记录见各`logs/gpu_smoke_attempt*/`。最终GPU验收状态由本目录的验收报告记录，不能仅凭CPU通过判断训练可用。

在已安排GPU资源并具有可靠收尾监督时，训练命令为：

```bash
cd /home/lvnuoyan/RAGEN2
bash run_factory.sh train.py --config-name qwen3_4b_nothink_smoke
```

正式训练需要选定旧实验配置、GPU及预算，并同时传两个False开关。`config/qwen3_4b_nothink.yaml`提供沿用旧默认值的Qwen3入口；旧实验配置也可覆盖模型路径及两个False开关。旧环境instruction中的CoT示例应在具体实验开始前单独审阅；本轮仅做上述用户批准的Qwen3指令修正，没有批量重写其他游戏提示词，也没有启动长训练。

人工review建议顺序：`config/qwen3_4b_nothink_smoke.yaml` → `ragen/llm_agent/ctx_manager.py` 的 `_render_chat_template/get_lm_inputs` → `ragen/trainer/agent_trainer.py` 的 `compute_advantage/fit` → `ragen/trainer/core_algos.py` → `ragen/workers/fsdp_workers.py` → 轨迹审计JSON。仓库内API密钥已按用户明确授权仅在服务器复制，不纳入本地快照或源码归档。

来源：[官方RAGEN安装脚本](https://github.com/mll-lab-nu/RAGEN/blob/9d021fc782a4a9e8e047f13c1cd706b32f16d8f2/scripts/setup_ragen.sh)、[官方worker迁移](https://github.com/mll-lab-nu/RAGEN/blob/9d021fc782a4a9e8e047f13c1cd706b32f16d8f2/ragen/workers/fsdp_workers.py)、[verl固定版本](https://github.com/verl-project/verl/tree/d62da4950573d7a4b7ef2362337952e7ab59e78d)。

## 2026-11-17 最终验收

最终运行：`logs/gpu_smoke_attempt4/`，两步GRPO退出码0，16条实际轨迹全部通过独立模板/token/mask/奖励复算。旧严格parser合法动作依次2/8、8/8；第一步归一化奖励与优势非零，PG loss=-0.0071788579；第二步同组回报相同，归一化优势为0，这是旧state归一化的预期。两步参数抽样分别37255/37888、37472/37888项改变，最大变化约2.03e-6、1.99e-6，均有限。

step2模型、scheduler/RNG和HF元数据已保存，scheduler last_epoch=2；模型checkpoint可读取，399个state tensors。已用新版`python -m verl.model_merger merge`完成CPU导出，目录`logs/gpu_smoke_attempt4/hf_model`。导出索引/分片匹配，原生chat_template和generation_config（含EOS、温度/top-p/top-k默认值）保持一致。未再次加载HF导出模型进行GPU生成；实际GPU生成来自训练中的vLLM权重同步。

可直接使用的入口：

```bash
# 只解析某个已有实验配置，核对实际参数；不启动训练。
bash run_qwen3_rl.sh _29_nmt_batch --cfg job --resolve

# 将一个新训练checkpoint导出为HF；目标目录须尚不存在。
bash merge_hf.sh FSDP_ACTOR_DIRECTORY NEW_HF_DIRECTORY
```

`run_qwen3_rl.sh`保留所选旧实验的算法/环境/预算，固定本轮非思考协议；可通过`RAGEN_MODEL_PATH`选择同系HF模型或checkpoint。正式训练仍需明确GPU分配、游戏配置与预算。

占卡最终由tmux `ragen2_qwen3_gpu4_20261117`监督，GPU4 UUID `GPU-61be0e3d-c360-3047-a51c-de3301fef717`，Qwen2.5-7B-Instruct、显存比例0.90、仅127.0.0.1:18440。独立实测health/models/4token推理均HTTP200，显存72073MiB（空闲0%计算利用率正常）。状态与健康证据在最终run的`status.json`和`verification.json`；长期监督器采用身份绑定的guard移交文件，后续不要直接并行启动新服务。

验收边界：本轮GPU覆盖单卡全参数FSDP1、单轮本地Bandit、同步GRPO；多轮模板有CPU双轮验证，但多轮游戏、多卡、PPO critic、LoRA、PCGrad/CST没有本轮GPU验收。两步smoke不保存optimizer状态，不作为正式训练checkpoint或研究效果结论。API密钥只在服务器按用户授权复制。
