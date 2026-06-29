# 验证 · 数据级模型验证

按数据集 **ID** 抽取「图片 + Prompt」，并对比不同模型 / 检查点的输出 —— 从数据层面验证模型效果。

本目录两个工具：
- `validate_by_id.ipynb` —— **交互逐样本**：挑一个 ID，看图片 + Prompt，逐模型对比输出（见下文）。
- `compare_bbox_prompt.py` —— **批量评测**：vLLM 部署两个模型（各 4 卡），在 saliency-r1-8k 上跑
  「给 / 不给 GT bounding box 提示」两种 prompt，算最终正确率。

---

## compare_bbox_prompt.py（vLLM 双模型 A/B 测评）

一条命令在 8 卡上**并发部署两个模型**（每模型 `tensor_parallel_size=4`），各自跑两种 prompt，最后汇总正确率：

- **方案① plain**：原始图片 + 问题。
- **方案② bbox**：同样的图片 + 问题，外加一句**鼓励使用**的英文提示，把 GT 框坐标填进去
  （“Pay special attention to the region inside the bounding box [...], evidence is there”）——**允许模型讲框/推理**。
- **方案③ noverbalize**：同样把 GT 框给模型，但用 OPD 真正的 **no-verbalize** 模板
  （baseline/hint 的 HINT_TEMPLATE：“use this only to decide where to look … **do NOT mention the box**”）——
  这才是 OPD 能蒸馏的"沉默 hint"。bbox vs noverbalize 的差 = "讲出来用" 带来的增益。
- （可选 **draw**：把框直接画在图上。）

三模型(两 teacher + 一 student)× 三方案,占满 8 卡(TP 4/2/2),先 rule 存盘:
```bash
M=/home/web_server/antispam/project/houshihao/models
D=/home/web_server/antispam/project/houshihao/datasets
uv run python 验证/compare_bbox_prompt.py \
    --models qwen3vl-8b=$M/Qwen3-VL-8B-Instruct,capcurriculum-8b=$M/CapCurriculum-8B,student-2b=$M/Qwen3-VL-2B-Instruct \
    --gpu-groups "0,1,2,3;4,5;6,7" \
    --schemes plain,bbox,noverbalize \
    --dataset $D/saliency-r1-8k --subsets docvqa,textvqa,gqa,openimages --limit 200 \
    --output-dir eval_outputs/bbox_ab3 --grader rule
```
跑完再 LLM judge **同一批已存的 records**(不重跑生成,需 `DEEPSEEK_API_KEY`)：
```bash
uv run python 验证/compare_bbox_prompt.py --judge-only \
    --output-dir eval_outputs/bbox_ab3 --grader llm
# → summary_llm.json + records_llm.jsonl(rule 版 summary.json/records.jsonl 保留)
```

（以上两个模型已是脚本默认值，直接 `uv run python 验证/compare_bbox_prompt.py` 也行。）

- `--limit` 是**每个子集**的样本上限；去掉 `--subsets`/`--limit` 跑全量。默认贪心解码。
- 产出：`eval_outputs/bbox_ab/summary.json`（每模型 × 每方案的正确率 + `Δ(bbox)`）、`records.jsonl`
  （逐样本：图片对应的 prompt、模型输出、抽取答案、对错）、`worker_*.log`（各模型 vLLM 日志，可 `tail -f`）。
- **判分**：默认 `--grader rule`（mathruler + 归一化精确匹配，无需 API）。**这两个是非 OPD 的 base 指令模型，
  不一定会用 `\boxed{}` 包答案**，rule 判分可能低估；建议加 `--grader llm`（DeepSeek 评判，需 `DEEPSEEK_API_KEY`，
  对 DocVQA 自由文本更准）。无论哪种，`Δ(bbox)` 都有意义（两方案偏差对称）。
- `--coord-mode normalized`(默认，0–1 坐标) / `pixel`；`--bbox-hint` 可改提示语模板。
- **内存安全**：按 `--batch-size`(默认 64)**分块**解码+生成，图片即用即放，RAM 不随数据量增长（全量 8K 也稳）；
  `--max-image-side`(默认 1536) 把超大图(如 DocVQA 扫描件)降采样，进一步压低 RAM/预处理开销。记录是**逐块写盘**的，
  中途挂了也保留已完成部分。若手动中断了主程序，记得 `nvidia-smi` 看一眼并 `pkill -f compare_bbox_prompt` 清理残留 worker。

---

## validate_by_id.ipynb（交互逐样本）

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
