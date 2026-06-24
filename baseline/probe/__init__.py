"""Evidence-reliance probe for the OPD project (the "命门" go/no-go experiment).

Stage 0 (no training) measures, for any VLM, whether its accuracy *causally*
depends on the GT evidence region of an image:

  Reliance = (Acc_full - Acc_mask_evidence) - (Acc_full - Acc_mask_random)
           =  Acc_mask_random - Acc_mask_evidence

i.e. how much more masking the *evidence* box hurts than masking a *random
equal-shape* box (the random mask cancels the generic "image is corrupted" OOD
artifact). Reliance >> 0 means the model really uses that region; Reliance ~ 0
means it is taking a shortcut / using priors.

  Delta_RG = Acc_crop - Acc_full   (region-grounding gap: how much handing the
             model the evidence crop helps over the full image)

All edits are pure pixel-space ops on the dataset's normalized bbox, so the
go/no-go decision does not depend on any model-internal hook or token mapping.
"""
