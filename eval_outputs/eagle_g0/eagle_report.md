# EAGLE-G0 — faithful causal attribution report

_correctness source: **llm_judge** · 2911 records across 5 models_

## Table 1 — region accuracy (plain)

| model | n | acc | IoU_EAGLE | point | area | suff | nec | vis_frac | IoU_LH | IoU_LH_boxed | SalR1_mass | SalR1_IoU |
|-------|---|-----|-----------|-------|------|------|-----|----------|--------|--------------|------------|-----------|
| qwen3vl-8b | 286 | 0.605 | 0.117 | 0.143 | 0.518 | 0.941 | 0.905 | 0.032 | — | — | — | — |
| capcurriculum-8b | 286 | 0.654 | 0.116 | 0.140 | 0.504 | 0.871 | 0.750 | 0.114 | — | — | — | — |
| qwen3vl-2b | 309 | 0.524 | 0.110 | 0.159 | 0.514 | 0.886 | 0.840 | 0.038 | — | — | — | — |
| qwen3vl-2b-opd | 285 | 0.593 | 0.112 | 0.123 | 0.509 | 0.895 | 0.820 | 0.031 | — | — | — | — |
| qwen3vl-2b-hint | 290 | 0.693 | 0.104 | 0.131 | 0.521 | 0.938 | 0.847 | 0.046 | — | — | — | — |

_IoU_EAGLE ≫ IoU_LH ⇒ LH attention under-measured localization (causal region is better). sufficiency=insertion AUC (↑ better), necessity=deletion AUC (↓ ⇒ region is necessary). SalR1 = Saliency-R1 map (mass@GT / top-20% IoU) — secondary attribution baseline._

## Group averages (plain) — first-5 (primary local-evidence) vs all-10

_macro = mean over per-subset means (each task type counts equally); pooled = size-weighted mean over records (dominated by big subsets gqa/flickr30k). **Prefer macro** — subset sizes differ ~25× (gqa 1765 vs vsr 70)._

### first-5 (gqa/openimages/v7w/textvqa/vsr) average

| model | #sub | n | acc | IoU_EAGLE | point | vis_log_lift | vis_reliance | suff | nec |
|-------|------|---|-----|-----------|-------|--------------|--------------|------|-----|
| qwen3vl-8b | 5 | 286 | 0.609 (0.605) | 0.120 (0.117) | 0.146 (0.143) | 0.082 (0.082) | 0.027 (0.027) | 0.941 (0.941) | 0.905 (0.905) |
| capcurriculum-8b | 5 | 286 | 0.657 (0.654) | 0.119 (0.116) | 0.144 (0.140) | 0.271 (0.265) | 0.100 (0.096) | 0.865 (0.871) | 0.743 (0.750) |
| qwen3vl-2b | 5 | 309 | 0.541 (0.524) | 0.113 (0.110) | 0.158 (0.159) | 0.081 (0.081) | 0.029 (0.030) | 0.887 (0.886) | 0.840 (0.840) |
| qwen3vl-2b-opd | 5 | 285 | 0.601 (0.593) | 0.116 (0.112) | 0.131 (0.123) | 0.075 (0.079) | 0.027 (0.028) | 0.892 (0.895) | 0.817 (0.820) |
| qwen3vl-2b-hint | 5 | 290 | 0.697 (0.693) | 0.108 (0.104) | 0.144 (0.131) | 0.135 (0.137) | 0.042 (0.041) | 0.936 (0.938) | 0.846 (0.847) |

### all-10 average

| model | #sub | n | acc | IoU_EAGLE | point | vis_log_lift | vis_reliance | suff | nec |
|-------|------|---|-----|-----------|-------|--------------|--------------|------|-----|
| qwen3vl-8b | 5 | 286 | 0.609 (0.605) | 0.120 (0.117) | 0.146 (0.143) | 0.082 (0.082) | 0.027 (0.027) | 0.941 (0.941) | 0.905 (0.905) |
| capcurriculum-8b | 5 | 286 | 0.657 (0.654) | 0.119 (0.116) | 0.144 (0.140) | 0.271 (0.265) | 0.100 (0.096) | 0.865 (0.871) | 0.743 (0.750) |
| qwen3vl-2b | 5 | 309 | 0.541 (0.524) | 0.113 (0.110) | 0.158 (0.159) | 0.081 (0.081) | 0.029 (0.030) | 0.887 (0.886) | 0.840 (0.840) |
| qwen3vl-2b-opd | 5 | 285 | 0.601 (0.593) | 0.116 (0.112) | 0.131 (0.123) | 0.075 (0.079) | 0.027 (0.028) | 0.892 (0.895) | 0.817 (0.820) |
| qwen3vl-2b-hint | 5 | 290 | 0.697 (0.693) | 0.108 (0.104) | 0.144 (0.131) | 0.135 (0.137) | 0.042 (0.041) | 0.936 (0.938) | 0.846 (0.847) |

_cells are `macro (pooled)`._

## Table 2 — looking vs using (per model, plain)

| model | n | acc | corr(c,IoU_EAGLE) | corr(c,vlog) | corr(c,vfrac) | IoU r/w | verdict |
|-------|---|-----|-------------------|--------------|---------------|---------|---------|
| qwen3vl-8b | 286 | 0.605 | -0.027 | -0.078 | -0.086 | 0.114/0.121 | mixed (corr(correct,IoU_EAGLE)=-0.03, corr(correct,visual_log_lift)=-0.08) |
| capcurriculum-8b | 286 | 0.654 | -0.039 | 0.055 | 0.132 | 0.112/0.122 | using bottleneck |
| qwen3vl-2b | 309 | 0.524 | -0.096 | -0.038 | 0.003 | 0.099/0.123 | mixed (corr(correct,IoU_EAGLE)=-0.10, corr(correct,visual_log_lift)=-0.04) |
| qwen3vl-2b-opd | 285 | 0.593 | -0.121 | 0.041 | 0.019 | 0.099/0.130 | mixed (corr(correct,IoU_EAGLE)=-0.12, corr(correct,visual_log_lift)=+0.04) |
| qwen3vl-2b-hint | 290 | 0.693 | -0.057 | 0.074 | 0.056 | 0.100/0.115 | mixed (corr(correct,IoU_EAGLE)=-0.06, corr(correct,visual_log_lift)=+0.07) |

- **qwen3vl-8b**: mixed (corr(correct,IoU_EAGLE)=-0.03, corr(correct,visual_log_lift)=-0.08)
- **capcurriculum-8b**: using bottleneck: IoU_EAGLE corr≈0 (-0.04) but visual-reliance corr=+0.13 (log-lift) → image-reliance, not localization, predicts correctness (output-level)
- **qwen3vl-2b**: mixed (corr(correct,IoU_EAGLE)=-0.10, corr(correct,visual_log_lift)=-0.04)
- **qwen3vl-2b-opd**: mixed (corr(correct,IoU_EAGLE)=-0.12, corr(correct,visual_log_lift)=+0.04)
- **qwen3vl-2b-hint**: mixed (corr(correct,IoU_EAGLE)=-0.06, corr(correct,visual_log_lift)=+0.07)

## Table 2b — looking vs using by task type (plain)

**qwen3vl-8b**

| subset | n | acc | corr(c,IoU_EAGLE) | corr(c,vlog) |
|--------|---|-----|-------------------|--------------|
| gqa | 64 | 0.547 | 0.185 | -0.113 |
| openimages | 60 | 0.317 | 0.257 | 0.050 |
| textvqa | 60 | 0.867 | 0.051 | 0.175 |
| v7w | 60 | 0.650 | -0.186 | -0.025 |
| vsr | 42 | 0.667 | -0.006 | -0.158 |

**capcurriculum-8b**

| subset | n | acc | corr(c,IoU_EAGLE) | corr(c,vlog) |
|--------|---|-----|-------------------|--------------|
| gqa | 64 | 0.578 | -0.021 | -0.082 |
| openimages | 60 | 0.400 | 0.261 | -0.046 |
| textvqa | 60 | 0.917 | 0.088 | 0.056 |
| v7w | 60 | 0.700 | 0.110 | 0.121 |
| vsr | 42 | 0.690 | -0.125 | 0.341 |

**qwen3vl-2b**

| subset | n | acc | corr(c,IoU_EAGLE) | corr(c,vlog) |
|--------|---|-----|-------------------|--------------|
| gqa | 87 | 0.391 | -0.111 | 0.086 |
| openimages | 60 | 0.283 | 0.343 | 0.109 |
| textvqa | 60 | 0.867 | -0.160 | 0.026 |
| v7w | 60 | 0.567 | -0.170 | -0.134 |
| vsr | 42 | 0.595 | -0.039 | -0.120 |

**qwen3vl-2b-opd**

| subset | n | acc | corr(c,IoU_EAGLE) | corr(c,vlog) |
|--------|---|-----|-------------------|--------------|
| gqa | 63 | 0.508 | -0.082 | 0.245 |
| openimages | 60 | 0.283 | 0.045 | -0.131 |
| textvqa | 60 | 0.883 | -0.002 | 0.121 |
| v7w | 60 | 0.617 | -0.401 | -0.039 |
| vsr | 42 | 0.714 | 0.181 | -0.080 |

**qwen3vl-2b-hint**

| subset | n | acc | corr(c,IoU_EAGLE) | corr(c,vlog) |
|--------|---|-----|-------------------|--------------|
| gqa | 68 | 0.647 | -0.031 | 0.094 |
| openimages | 60 | 0.417 | 0.223 | 0.170 |
| textvqa | 60 | 0.950 | 0.082 | 0.025 |
| v7w | 60 | 0.733 | -0.288 | -0.004 |
| vsr | 42 | 0.738 | 0.057 | -0.045 |

## Table 3 — hint mechanism (plain vs hint, paired)
- **qwen3vl-8b** [generate]: n=282, Δacc=0.050 (plain 0.606→hint 0.656), ΔIoU_EAGLE=0.006, Δvisual_reliance=0.003 → output-level: hint helps with ~unchanged causal region (likely text-routing on the hint coords)
- **capcurriculum-8b** [generate]: n=282, Δacc=0.053 (plain 0.656→hint 0.709), ΔIoU_EAGLE=-0.003, Δvisual_reliance=-0.017 → output-level: hint helps with ~unchanged causal region (likely text-routing on the hint coords)
- **qwen3vl-2b** [generate]: n=282, Δacc=0.050 (plain 0.539→hint 0.589), ΔIoU_EAGLE=0.001, Δvisual_reliance=0.019 → output-level: hint helps with ~unchanged causal region (likely text-routing on the hint coords)
- **qwen3vl-2b-opd** [generate]: n=282, Δacc=0.074 (plain 0.596→hint 0.670), ΔIoU_EAGLE=-0.004, Δvisual_reliance=0.020 → output-level: hint helps with ~unchanged causal region (likely text-routing on the hint coords)
- **qwen3vl-2b-hint** [generate]: n=282, Δacc=-0.039 (plain 0.695→hint 0.656), ΔIoU_EAGLE=0.006, Δvisual_reliance=0.026 → no effect

## Table 4 — OPD / hint training raises image-reliance? (plain)

| role | model | n | acc | visual_log_lift | visual_reliance | visual_fraction | IoU_EAGLE | necessity |
|------|-------|---|-----|-----------------|-----------------|-----------------|-----------|-----------|
| base | qwen3vl-2b | 309 | 0.524 | 0.081 | 0.030 | 0.038 | 0.110 | 0.840 |
| opd | qwen3vl-2b-opd | 285 | 0.593 | 0.079 | 0.028 | 0.031 | 0.112 | 0.820 |
| hint_opd | qwen3vl-2b-hint | 290 | 0.693 | 0.137 | 0.041 | 0.046 | 0.104 | 0.847 |

_Success story = training ↑ accuracy AND ↑ visual_log_lift / visual_reliance (answer depends on the image more), even if IoU_EAGLE barely moves._

