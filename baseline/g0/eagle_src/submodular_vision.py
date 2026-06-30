"""Vendored verbatim from EAGLE (interpretation/submodular_vision.py).

EAGLE, arXiv 2509.22496, https://github.com/RuoyuChen10/EAGLE (MIT License).
Base submodular subset-selection explainer. Unused imports (cv2/time/threadpool)
dropped so this imports without opencv; the algorithm is unchanged.
"""

import numpy as np
import torch
from tqdm import tqdm


class MLLMSubModularExplanationVision(object):
    """
    Black-box explanation of multimodal large language
    model (MLLM) based on submodular subset selection.
    """
    def __init__(self,
                 model,
                 preproccessing_function=None,
                 lambda1=1.0,
                 lambda2=1.0,
                 ):
        super(MLLMSubModularExplanationVision, self).__init__()
        # Parameters of the submodular
        self.MLLM = model
        self.preproccessing_function = preproccessing_function

        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.device = self.MLLM.device

    def save_file_init(self):
        self.saved_json_file = {}
        self.saved_json_file["insertion_score"] = []
        self.saved_json_file["deletion_score"] = []
        self.saved_json_file["smdl_score"] = []
        self.saved_json_file["insertion_word_score"] = []
        self.saved_json_file["deletion_word_score"] = []
        self.saved_json_file["region_area"] = []
        self.saved_json_file["lambda1"] = self.lambda1
        self.saved_json_file["lambda2"] = self.lambda2

    def MLLM_inference_batch_images(self, images):
        results = []
        for image in images:
            output_logits = self.MLLM(image)
            results.append(output_logits)

        results = torch.stack(results, dim=0)
        return results

    def evaluation_maximun_sample(self, S_set):
        V_set_tensor = torch.from_numpy(np.array(self.V_set)).float().to(self.device)

        alpha_batch = V_set_tensor + self.refer_baseline.unsqueeze(0)
        alpha_batch = alpha_batch.expand(-1, -1, -1, 3)

        source_tensor = self.source_tensor.unsqueeze(0).expand(alpha_batch.shape[0], -1, -1, -1)
        batch_input_images = alpha_batch * source_tensor
        batch_input_images_reverse = (1 - alpha_batch) * source_tensor

        with torch.no_grad():
            # Insertion
            insertion_scores = self.MLLM_inference_batch_images(batch_input_images).to(torch.float32)

            # Deletion
            deletion_scores = self.MLLM_inference_batch_images(batch_input_images_reverse).to(torch.float32)

            # Overall submodular score
            smdl_scores = self.lambda1 * insertion_scores + self.lambda2 * (1 - deletion_scores)
            smdl_scores = smdl_scores.mean(-1)
            arg_max_index = smdl_scores.argmax().cpu().item()

            # Save intermediate results
            self.saved_json_file["insertion_score"].append(insertion_scores[arg_max_index].mean().cpu().numpy().item())
            self.saved_json_file["insertion_word_score"].append(insertion_scores[arg_max_index].cpu().numpy().tolist())

            self.saved_json_file["deletion_score"].append(deletion_scores[arg_max_index].mean().cpu().numpy().item())
            self.saved_json_file["deletion_word_score"].append(deletion_scores[arg_max_index].cpu().numpy().tolist())

            self.saved_json_file["smdl_score"].append(smdl_scores[arg_max_index].cpu().item())

            # Update
            S_set.append(self.V_set[arg_max_index])
            self.refer_baseline = self.refer_baseline + torch.from_numpy(self.V_set[arg_max_index]).float().to(self.device)
            del self.V_set[arg_max_index]

            self.saved_json_file["region_area"].append(
                (self.refer_baseline.sum() / self.region_area).cpu().item()
            )

        return S_set

    def get_merge_set(self):
        # define a subset
        S_set = []
        self.refer_baseline = torch.zeros_like(torch.from_numpy(self.V_set[0]).float(), device=self.device)

        for i in tqdm(range(self.saved_json_file["sub-region_number"])):
            S_set = self.evaluation_maximun_sample(S_set)

        self.saved_json_file["org_score"] = self.saved_json_file["insertion_word_score"][-1]
        self.saved_json_file["baseline_score"] = self.saved_json_file["deletion_word_score"][-1]

        return S_set

    def __call__(self, image, V_set):
        self.save_file_init()
        self.saved_json_file["sub-region_number"] = len(V_set)

        self.source_image = image
        self.source_tensor = torch.from_numpy(self.source_image).float().to(self.device)
        self.h, self.w, _ = self.source_image.shape
        self.region_area = image.shape[0] * image.shape[1]

        self.V_set = V_set.copy()

        Submodular_Subset = self.get_merge_set()

        return Submodular_Subset, self.saved_json_file
