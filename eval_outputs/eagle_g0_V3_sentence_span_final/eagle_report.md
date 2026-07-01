# EAGLE-G0 — faithful causal attribution report

_correctness source: **llm_judge** · spatial metric source: **record** · 3179 records across 4 models_

## Table 1 — region accuracy (plain)

| model | n | acc | IoU_EAGLE | Point@1 | Energy | IoU@10 | IoU@20 | DelDrop | InsRec | vis_frac |
|-------|---|-----|-----------|---------|--------|--------|--------|---------|--------|----------|
| capcurriculum-8b | 241 | 0.701 | 0.152 | 0.207 | 0.168 | 0.139 | 0.151 | 0.518 | 0.390 | 0.191 |
| qwen3vl-2b | 277 | 0.668 | 0.158 | 0.177 | 0.174 | 0.124 | 0.152 | 0.308 | 0.357 | 0.146 |
| qwen3vl-2b-opd | 268 | 0.649 | 0.153 | 0.172 | 0.174 | 0.129 | 0.149 | 0.418 | 0.473 | 0.184 |
| qwen3vl-2b-hiddenhint-opd | 280 | 0.693 | 0.146 | 0.207 | 0.161 | 0.125 | 0.142 | 0.524 | 0.623 | 0.209 |

_Point@1 = max heat patch in GT. Energy = heatmap mass inside GT. IoU@10/20 threshold the top 10%/20% area of the final aggregate map. DelDrop/InsRec are target-span logp drop/recovery after deleting/keeping top-20% attributed area._

## Group averages (plain) — first-5 (primary local-evidence) vs all-10

_macro = mean over per-subset means (each task type counts equally); pooled = size-weighted mean over records (dominated by big subsets gqa/flickr30k). **Prefer macro** — subset sizes differ ~25× (gqa 1765 vs vsr 70)._

### first-5 (gqa/openimages/v7w/textvqa/vsr) average

| model | #sub | n | acc | IoU_EAGLE | Point@1 | Energy | IoU@20 | DelDrop | InsRec | vis_log_lift |
|-------|------|---|-----|-----------|---------|--------|--------|---------|--------|--------------|
| capcurriculum-8b | 5 | 241 | 0.698 (0.701) | 0.158 (0.152) | 0.215 (0.207) | 0.175 (0.168) | 0.153 (0.151) | 0.502 (0.518) | 0.375 (0.390) | 0.723 (0.737) |
| qwen3vl-2b | 5 | 277 | 0.672 (0.668) | 0.162 (0.158) | 0.181 (0.177) | 0.180 (0.174) | 0.155 (0.152) | 0.302 (0.308) | 0.350 (0.357) | 0.555 (0.560) |
| qwen3vl-2b-opd | 5 | 268 | 0.650 (0.649) | 0.159 (0.153) | 0.180 (0.172) | 0.182 (0.174) | 0.154 (0.149) | 0.405 (0.418) | 0.460 (0.473) | 0.749 (0.760) |
| qwen3vl-2b-hiddenhint-opd | 5 | 280 | 0.696 (0.693) | 0.150 (0.146) | 0.209 (0.207) | 0.168 (0.161) | 0.145 (0.142) | 0.512 (0.524) | 0.610 (0.623) | 0.912 (0.921) |

### all-10 average

| model | #sub | n | acc | IoU_EAGLE | Point@1 | Energy | IoU@20 | DelDrop | InsRec | vis_log_lift |
|-------|------|---|-----|-----------|---------|--------|--------|---------|--------|--------------|
| capcurriculum-8b | 5 | 241 | 0.698 (0.701) | 0.158 (0.152) | 0.215 (0.207) | 0.175 (0.168) | 0.153 (0.151) | 0.502 (0.518) | 0.375 (0.390) | 0.723 (0.737) |
| qwen3vl-2b | 5 | 277 | 0.672 (0.668) | 0.162 (0.158) | 0.181 (0.177) | 0.180 (0.174) | 0.155 (0.152) | 0.302 (0.308) | 0.350 (0.357) | 0.555 (0.560) |
| qwen3vl-2b-opd | 5 | 268 | 0.650 (0.649) | 0.159 (0.153) | 0.180 (0.172) | 0.182 (0.174) | 0.154 (0.149) | 0.405 (0.418) | 0.460 (0.473) | 0.749 (0.760) |
| qwen3vl-2b-hiddenhint-opd | 5 | 280 | 0.696 (0.693) | 0.150 (0.146) | 0.209 (0.207) | 0.168 (0.161) | 0.145 (0.142) | 0.512 (0.524) | 0.610 (0.623) | 0.912 (0.921) |

_cells are `macro (pooled)`._

## Table 2 — looking vs using (per model, plain)

| model | n | acc | corr(c,IoU_EAGLE) | corr(c,vlog) | corr(c,vfrac) | IoU r/w | verdict |
|-------|---|-----|-------------------|--------------|---------------|---------|---------|
| capcurriculum-8b | 241 | 0.701 | -0.003 | 0.151 | 0.161 | 0.152/0.153 | using bottleneck |
| qwen3vl-2b | 277 | 0.668 | -0.059 | 0.154 | 0.167 | 0.152/0.170 | using bottleneck |
| qwen3vl-2b-opd | 268 | 0.649 | -0.027 | 0.158 | 0.183 | 0.150/0.159 | using bottleneck |
| qwen3vl-2b-hiddenhint-opd | 280 | 0.693 | 0.013 | 0.089 | 0.071 | 0.147/0.143 | mixed (corr(correct,IoU_EAGLE)=+0.01, corr(correct,visual_log_lift)=+0.09) |

- **capcurriculum-8b**: using bottleneck: IoU_EAGLE corr≈0 (-0.00) but visual-reliance corr=+0.16 (log-lift) → image-reliance, not localization, predicts correctness (output-level)
- **qwen3vl-2b**: using bottleneck: IoU_EAGLE corr≈0 (-0.06) but visual-reliance corr=+0.17 (log-lift) → image-reliance, not localization, predicts correctness (output-level)
- **qwen3vl-2b-opd**: using bottleneck: IoU_EAGLE corr≈0 (-0.03) but visual-reliance corr=+0.18 (log-lift) → image-reliance, not localization, predicts correctness (output-level)
- **qwen3vl-2b-hiddenhint-opd**: mixed (corr(correct,IoU_EAGLE)=+0.01, corr(correct,visual_log_lift)=+0.09)

## Table 2b — looking vs using by task type (plain)

**capcurriculum-8b**

| subset | n | acc | corr(c,IoU_EAGLE) | corr(c,vlog) |
|--------|---|-----|-------------------|--------------|
| gqa | 50 | 0.640 | -0.000 | -0.002 |
| openimages | 50 | 0.440 | 0.370 | 0.031 |
| textvqa | 57 | 0.930 | 0.007 | 0.033 |
| v7w | 45 | 0.711 | -0.081 | 0.052 |
| vsr | 39 | 0.769 | 0.047 | 0.222 |

**qwen3vl-2b**

| subset | n | acc | corr(c,IoU_EAGLE) | corr(c,vlog) |
|--------|---|-----|-------------------|--------------|
| gqa | 58 | 0.638 | 0.033 | 0.070 |
| openimages | 59 | 0.339 | 0.203 | 0.032 |
| textvqa | 60 | 0.933 | -0.039 | 0.050 |
| v7w | 58 | 0.690 | -0.162 | 0.040 |
| vsr | 42 | 0.762 | -0.104 | -0.122 |

**qwen3vl-2b-opd**

| subset | n | acc | corr(c,IoU_EAGLE) | corr(c,vlog) |
|--------|---|-----|-------------------|--------------|
| gqa | 59 | 0.610 | 0.167 | 0.047 |
| openimages | 54 | 0.315 | 0.131 | -0.058 |
| textvqa | 60 | 0.900 | -0.068 | 0.258 |
| v7w | 55 | 0.673 | -0.190 | -0.049 |
| vsr | 40 | 0.750 | 0.097 | 0.140 |

**qwen3vl-2b-hiddenhint-opd**

| subset | n | acc | corr(c,IoU_EAGLE) | corr(c,vlog) |
|--------|---|-----|-------------------|--------------|
| gqa | 60 | 0.650 | 0.110 | 0.015 |
| openimages | 59 | 0.373 | 0.299 | -0.153 |
| textvqa | 60 | 0.950 | 0.146 | 0.152 |
| v7w | 60 | 0.750 | -0.135 | -0.254 |
| vsr | 41 | 0.756 | 0.090 | -0.024 |

## Table 3 — hint mechanism (plain vs hint, paired)
- **capcurriculum-8b** [generate]: n=194, Δacc=0.041 (plain 0.716→hint 0.758), ΔIoU_EAGLE=0.008, Δvisual_reliance=-0.054 → output-level: hint helps with ~unchanged causal region (likely text-routing on the hint coords)
- **qwen3vl-2b** [generate]: n=268, Δacc=-0.015 (plain 0.683→hint 0.668), ΔIoU_EAGLE=0.001, Δvisual_reliance=-0.027 → no effect
- **qwen3vl-2b-opd** [generate]: n=239, Δacc=0.008 (plain 0.678→hint 0.686), ΔIoU_EAGLE=0.005, Δvisual_reliance=-0.052 → no effect
- **qwen3vl-2b-hiddenhint-opd** [generate]: n=280, Δacc=-0.046 (plain 0.693→hint 0.646), ΔIoU_EAGLE=0.007, Δvisual_reliance=-0.059 → no effect

## Table 4 — OPD / hint training raises image-reliance? (plain)

| role | model | n | acc | visual_log_lift | visual_reliance | visual_fraction | IoU_EAGLE | necessity |
|------|-------|---|-----|-----------------|-----------------|-----------------|-----------|-----------|
| base | qwen3vl-2b | 277 | 0.668 | 0.560 | 0.118 | 0.146 | 0.158 | — |
| opd | qwen3vl-2b-opd | 268 | 0.649 | 0.760 | 0.150 | 0.184 | 0.153 | — |
| hint_opd | qwen3vl-2b-hiddenhint-opd | 280 | 0.693 | 0.921 | 0.174 | 0.209 | 0.146 | — |

_Success story = training ↑ accuracy AND ↑ visual_log_lift / visual_reliance (answer depends on the image more), even if IoU_EAGLE barely moves._
