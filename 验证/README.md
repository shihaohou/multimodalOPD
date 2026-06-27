# 验证 · 数据级模型验证

按数据集 **ID** 抽取「图片 + Prompt」，并对比不同模型 / 检查点的输出 —— 从数据层面验证模型效果。

## 用法

在 GPU 机器上、用 OPD 的 uv 环境启动 Jupyter，打开 `validate_by_id.ipynb`：

```bash
cd <repo>            # 含 baseline/ 与 vigos/ 的仓库根目录
uv run jupyter lab   # 或 uv run jupyter notebook
```

notebook 按 ①→⑥ 顺序运行即可：

1. **① 配置** —— 通常只改这一格：`DATASET` / `MODELS` / `SYSTEM_PROMPT_STYLE` / 生成参数。
2. **② 环境** —— 定位仓库根目录、import、模型懒加载缓存。
3. **③ 加载数据** —— 建立 `ID -> 样本` 索引，打印可用 ID。
4. **④ 指定 ID** —— 设 `SAMPLE_ID`，查看图片（含 GT 证据框）+ 模型实际看到的完整 Prompt。
5. **⑤ 模型输出** —— 单模型 / 多模型对比同一 ID 的输出、抽取答案、判分。
6. **⑥（可选）证据热力图** —— 模型答案 token 的显著性是否落在 GT 证据框 `bbox` 内（需 eager 注意力）。

## 默认数据集

`peterant330/saliency-r1-8k`（本地：`.../datasets/saliency-r1-8k`）。
样本字段：`question_id`(ID) · `problem` · `solution` · `image` · `bbox`(证据框) · `dataset`(子集)。

换成别的 OPD 数据集（本地目录或 HF id）也可以：会自动回退到通用加载器
（`problem`/`image`/`answer`，ID 取 `question_id`/`problem_id`/`id`/行号），但没有 `bbox` 与第⑥步热力图。

## 说明

- 复用仓库现成代码（`baseline/*`、`vigos/*`），Prompt / 抽答案 / 判分与训练、`scripts/eval_opd.sh` 一致。
- 本 notebook 用 HF transformers 交互推理，数值可能与 vLLM 评测略有差异；要对齐论文 / 榜单请用
  `scripts/eval_opd.sh` / `scripts/eval_suite.sh`。
- `SYSTEM_PROMPT_STYLE` 必须与训练时一致。
