from .prelude import *
from lpips import LPIPS


class MultiscaleLPIPS:
    def __init__(
        self,
        min_loss_res: int = 16,
        level_weights: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        l1_weight: float = 0.1
    ):
        super().__init__()
        self.min_loss_res = min_loss_res
        self.weights = level_weights
        self.l1_weight = l1_weight
        # Added spatial=True so it returns the (H, W) map instead of a single scalar
        self.lpips_network = LPIPS(net="vgg", spatial=True, verbose=False).cuda()

    def measure_lpips(self, x, y, mask, return_map=False):
        if mask is not None:
            noise = (torch.randn_like(x) + 0.5) / 2.0
            x = x + noise * (1.0 - mask)
            y = y + noise * (1.0 - mask)

    # Use the 'retPerLayer=False' but capture the raw spatial output if needed
    # Most LPIPS versions return (Batch, 1, H, W) if called directly
        dist_map = self.lpips_network(x, y, normalize=True)
    
        if return_map:
            return dist_map.mean(), dist_map.detach()
        return dist_map.mean()     
    def gram_matrix(self, x):
        (b, c, h, w) = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

    def __call__(self, f_hat, x_clean: Tensor, y: Tensor, mask: Optional[Tensor] = None):
        x = f_hat(x_clean)
        losses = []
        
        # Initialize the spatial map storage
        self.last_spatial_map = None

        if mask is not None:
            mask = F.interpolate(mask, y.shape[-1], mode="area")

        # Iterate through multiscale resolutions
        for i, weight in enumerate(self.weights):
            # At extremely low resolutions, LPIPS stops making sense
            if y.shape[-1] <= self.min_loss_res:
                break
            
            if weight > 0:
                # LFMA Modification: Capture the spatial map from the 
                # highest resolution (first iteration) to guide the W++ update.
                if i == 0:
                    loss, spatial_map = self.measure_lpips(x, y, mask, return_map=True)
                    self.last_spatial_map = spatial_map
                else:
                    loss = self.measure_lpips(x, y, mask)
                
                losses.append(weight * loss)

            # Downsample for the next multiscale level
            if mask is not None:
                mask = F.avg_pool2d(mask, 2)

            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
        
        # Aggregate multiscale losses
        total = torch.stack(losses).sum(dim=0) if len(losses) > 0 else 0.0
        
        # Final L1 loss at the lowest resolution reached
        l1 = self.l1_weight * F.l1_loss(x, y)

        return total + l1
