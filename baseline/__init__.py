"""Baselines for the multimodal OPD project, built on the ViGOS framework.

Currently contains vanilla On-Policy Distillation (OPD). Code here reuses the
``vigos`` package (rollout, teacher forward, KL, DDP loss normalization) but lives
outside it so the upstream ViGOS / OPSD code stays untouched.
"""
