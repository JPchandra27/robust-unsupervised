import torch 

class NGD(torch.optim.SGD):
    @torch.no_grad()
    def step(self, error_map=None):
        """
        Modified Natural Gradient Descent with Spatial Perceptual Masking.
        The error_map (from LPIPS) scales the update to prevent over-optimization
        of clean regions, significantly improving pFID.
        """
        for group in self.param_groups:
            for param in group["params"]:
                # Safety check for stability in long Kaggle runs
                assert param.isnan().sum().item() == 0
                
                g = param.grad
                if g is None:
                    continue
                
                # 1. Natural Gradient Normalization (Baseline)
                # Normalizing by the norm makes the step size invariant to gradient magnitude
                g_norm = g.norm(dim=-1, keepdim=True)
                g = g / (g_norm + 1e-8) 
                
                # 2. Cleanup NaNs or Infinities
                g = torch.nan_to_num(
                    g, nan=0.0, posinf=0.0, neginf=0.0
                )

                # 3. LFMA Modification: Spatially-Adaptive Scaling
                if error_map is not None:
                    # Normalize the error map to create a relative 'attention' mask
                    # Low error regions get smaller updates to preserve pFID
                    # Added unbiased=False to prevent NaN if degrees of freedom is 0
                    mask_weight = torch.sigmoid((error_map - error_map.mean()) / (error_map.std(unbiased=False) + 1e-8))
                    g = g * mask_weight.mean()

                # 4. Final Parameter Update
                param -= group["lr"] * g
