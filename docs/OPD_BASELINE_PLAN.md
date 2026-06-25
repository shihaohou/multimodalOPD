# OPD Baseline 实现方案（文件级）

目标：在现有 ViGOS（多模态 OPSD）仓库中**新增一条 On-Policy Distillation (OPD) 训练路径**，
与 ViGOS 并存。teacher = 一个**独立加载、冻结**的更强同族 VLM（ckpt 路径可配置，base 或 RL 过的均可），
**不喂参考答案**，对学生 on-policy rollout 的整段 completion 做**逐 token 反向 KL** `KL(student‖teacher)`。

> **实现状态（已落地，设计微调）**：最终采用**独立入口文件**而非在 `train_vigos.py`/`trainer.py` 内分流，
> 以保证 ViGOS 原文件零改动。实际代码位于独立的 `baseline/` 包（`vigos/` 仅作为库被复用）：
> `baseline/opd_data_collator.py`（`OPDDataCollator`）、`baseline/opd_trainer.py`（`OPDTrainer(ViGOSTrainer)` 仅覆写 `compute_loss`）、
> `baseline/train_opd.py`（入口）、`scripts/train_opd.sh`、`README_OPD.md`。
> 下面第 2、3 节描述的"改 `train_vigos.py`/`trainer.py`"为早期方案，逻辑等价、仅作参考；以独立文件为准。
> data_collator 也**未改动**——OPD 的 prompt 逻辑放在新的 `OPDDataCollator` 里（子类复用 `_encode`）。

## 0. 核心设计

| 维度 | OPSD（现状） | OPD（新增） |
|---|---|---|
| teacher 权重 | 关掉 LoRA 的 base model（同权重） | 独立加载的另一个 ckpt（`teacher_model_name_or_path`） |
| teacher prompt | 特权（含参考答案） | 和 student **完全相同**的非特权 prompt |
| 监督 token | description / think / answer 分段 | 整段 completion（`completion_attention_mask`） |
| loss | 3 项（λ_perc/reas/ref） | 1 项反向 KL（`λ_opd`） |
| teacher 前向 | `model` + `disable_adapter()` | `self.teacher_model` 直接前向（no_grad/eval） |

关键约束：**teacher 必须和 student 共享同一 tokenizer / 词表**（全词表精确 KL 要求 token 对齐），
同族 Qwen2.5-VL / Qwen3-VL 满足。`processor` 沿用 student 的即可。

## 1. `vigos/data_collator.py`

新增一个 plain-CoT 学生 prompt（不带 description 段），让 OPD baseline 与论文 OPSD baseline 公平对齐。

```python
def format_opd_student_prompt(problem: Any) -> str:
    problem_text = str(problem).strip()
    return (
        f"Problem: {problem_text}\n\n"
        "Reason step by step about the image and the question. "
        "Enclose your reasoning within <think> </think> tags.\n"
        "Finally, provide a single word or phrase answer in \\boxed{}.\n"
        "The output format should be: <think> reasoning process here </think> \\boxed{FINAL ANSWER here}."
    )
```

`ViGOSDataCollator` 增加字段 `prompt_style: str = "vigos"`。`__call__` 内：

- `prompt_style == "cot"` 时：
  - student message 用 `format_opd_student_prompt`，prefill 用 `THINK_PREFILL`（`<think>`）。
  - `student_prompt_texts` 用 `... + THINK_PREFILL`。
  - **跳过** perception / reasoning / reference 三个 message 的构建与 `_encode`（省显存和算力）。
  - 仍产出 `student_prompt_*`、`student_prompt_texts`、`student_images`、`vigos_problems/answers`、`sample_ids`。
- `prompt_style == "vigos"` 时：保持现状（也允许"OPD + ViGOS 格式"作为消融）。

## 2. `vigos/train_vigos.py`

`ViGOSScriptArguments` 增加：

```python
training_mode: str = "vigos"                 # "vigos" | "opd"
teacher_model_name_or_path: Optional[str] = None
teacher_torch_dtype: str = "bfloat16"
teacher_attn_implementation: str = "flash_attention_2"
student_prompt_style: str = "vigos"          # OPD 建议 "cot"
lambda_opd: float = 1.0
```

模型构建后、建 trainer 前：

```python
teacher_model = None
if script_args.training_mode == "opd":
    if not script_args.teacher_model_name_or_path:
        raise ValueError("OPD mode requires --teacher_model_name_or_path.")
    t_cls, _ = _model_class_for_checkpoint(script_args.teacher_model_name_or_path)
    teacher_model = t_cls.from_pretrained(
        script_args.teacher_model_name_or_path,
        torch_dtype=getattr(torch, script_args.teacher_torch_dtype),
        attn_implementation=script_args.teacher_attn_implementation,
    )
    teacher_model.config.use_cache = False
    teacher_model.requires_grad_(False)
    teacher_model.eval()
    # 不上 LoRA、不进 deepspeed/accelerate.prepare；仅放到本 rank 设备
    # （放置可在此处 .to(...)，或交给 trainer.__init__ 用 self.accelerator.device）
```

`ViGOSDataCollator(..., prompt_style=script_args.student_prompt_style)`，并把
`training_mode / teacher_model / lambda_opd` 传入 `ViGOSTrainer`。

## 3. `vigos/trainer.py`

### 3.1 `__init__`
新增参数 `training_mode="vigos"`、`teacher_model=None`、`lambda_opd=1.0`，保存为属性。
若 `teacher_model is not None`：`self.teacher_model = teacher_model.to(self.accelerator.device).eval()`，
`self.teacher_model.requires_grad_(False)`。（teacher **不**注册到 vLLM 同步回调。）

### 3.2 `compute_loss` 顶部分流
```python
if getattr(self, "training_mode", "vigos") == "opd":
    return self._compute_opd_loss(model, inputs, return_outputs)
```

### 3.3 新增 `_compute_opd_loss`（几乎全是复用现有 helper）
```python
def _compute_opd_loss(self, model, inputs, return_outputs=False):
    student_prompt = self._prompt_inputs(inputs, "student")
    rollout = self._generate_on_policy(model, student_prompt, inputs)
    self._maybe_log_completion_snapshot(inputs, rollout)  # 可选

    completion_ids = rollout["completion_ids"]
    completion_attention = rollout["completion_attention_mask"].to(dtype=torch.bool)

    # 学生前向（带梯度）
    student_inputs = self._with_completion(
        student_prompt,
        full_input_ids=rollout["generated_ids"],
        full_attention_mask=rollout["generated_attention_mask"],
    )
    student_inputs["logits_to_keep"] = completion_ids.shape[1] + 1
    student_outputs = model(**student_inputs)
    student_logits = self._completion_logits(student_outputs.logits, completion_ids.shape[1])
    del student_outputs

    # teacher 前向（独立模型，no_grad/eval，复用现有批处理函数）
    teacher_inputs = self._append_completion(
        student_prompt, completion_ids, rollout["completion_attention_mask"],
    )
    teacher_logits = self._batched_teacher_completion_logits(
        self.teacher_model,
        [{"name": "opd", "inputs": teacher_inputs,
          "completion_length": completion_ids.shape[1]}],
    )["opd"]

    # 反向 KL：source=student → KL(student‖teacher)
    opd_loss = masked_kl_loss(
        student_logits, teacher_logits, completion_attention,
        temperature=self.distill_temperature, token_clip=self.token_loss_clip,
    )
    opd_loss, _, num, cnt = self._distributed_masked_loss_with_stats(opd_loss, completion_attention)
    loss = self.lambda_opd * opd_loss

    # 指标（复用）
    correct = self._rollout_answer_correctness(inputs, rollout)
    _, ac, an = self._distributed_rate_stats(correct)
    self._record_loss_metrics({"loss_opd": (num, cnt), "answer_accuracy": (ac, an)})

    if return_outputs:
        return loss, {"logits": student_logits.detach()}
    return loss
```

复用要点（均已存在）：
- `_prompt_inputs / _generate_on_policy`（vLLM + HF 两条 rollout 路径都返回 `generated_ids` 等）
- `_with_completion / _append_completion / _completion_logits`
- `_batched_teacher_completion_logits(model, jobs)`：对非 PEFT 的独立 teacher，`_teacher_context` 自动退化为 `nullcontext`，内部已 `torch.no_grad()` + eval —— **直接传 `self.teacher_model` 即可**
- `masked_kl_loss(source, target, mask)` = `KL(source‖target)`（现有 ref 项已是 `masked_kl_loss(student, ref, ...)` 这种反向 KL 用法）
- `_distributed_masked_loss_with_stats`（DDP 全局归一化，**必须保留**）

## 4. 新增 `scripts/train_opd.sh`
基于 `scripts/train_vigos_qwen25_3b.sh` 复制，改动：
- 新增（必填）`TEACHER_MODEL="${TEACHER_MODEL:?Set TEACHER_MODEL ...}"`
- 传 `--training_mode opd --teacher_model_name_or_path "$TEACHER_MODEL" --student_prompt_style cot --lambda_opd "${LAMBDA_OPD:-1.0}"`
- 删去 `--lambda_perception/--lambda_reasoning/--lambda_ref/--description_last_token_clip/--reasoning_first_token_clip`（保留 `--token_loss_clip`、`--distill_temperature`）
- 给 teacher 腾显存：默认 `VLLM_GPU_MEMORY_UTILIZATION=0.30`

## 5. 显存预算与 teacher 规模

teacher 在每个 rank 上**复制**一份（只前向、无梯度/优化器）：
- 7B bf16 ≈ 14 GB/卡 —— 3B/7B 学生 + colocate vLLM（调低 util）在 A100-80G **可行**
- 14B ≈ 28 GB/卡 —— 紧张但可调（再降 vLLM util / per-device bs）
- ≥32B —— 单卡放不下，需要 teacher 张量并行（独立进程 / `device_map`）或单独占卡；不在本方案默认范围
- vLLM 的 logprob 接口只给 top-k，**不能**满足全词表 KL，因此 teacher 必须走本地 HF 前向

> 若坚持用 ≥32B teacher：要么改成 **top-k KL**（`losses.py` 已有 `js_tokens` 的 top-k gather 思路可借鉴），要么把 teacher 放到独立 GPU 组做 TP。这会显著增加复杂度，建议 baseline 先用 ≤14B。

## 6. 验证（5 步 smoke test）
```bash
DATASET_NAME=LMMs-Lab-Turtle/Vision-SR1-47K \
TEACHER_MODEL=Qwen/Qwen2.5-VL-7B-Instruct \
MAX_STEPS=5 SAVE_STEPS=5 REPORT_TO=none \
VLLM_GPU_MEMORY_UTILIZATION=0.30 \
bash scripts/train_opd.sh
```
检查：能跑通；`loss_opd` 为正且下降趋势；`answer_accuracy` 有输出；显存不 OOM。

## 7. 改动量
- 改 3 个文件（`data_collator.py` 约 +25 行、`train_vigos.py` 约 +20 行、`trainer.py` 约 +55 行）
- 新增 1 个脚本
- ViGOS 原路径零影响（全部走 `training_mode` 分支）
