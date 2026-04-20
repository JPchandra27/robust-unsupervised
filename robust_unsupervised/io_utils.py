from robust_unsupervised.prelude import *
from robust_unsupervised.variables import *

import shutil
import torch_utils as torch_utils
import torch_utils.misc as misc
import contextlib

import PIL.Image as Image


def open_generator(pkl_path: str, refresh=True, float=True, ema=True) -> networks.Generator:
    print(f"Loading generator from {pkl_path}...")

    with dnnlib.util.open_url(pkl_path) as fp:
        G = legacy.load_network_pkl(fp)["G_ema" if ema else "G"].cuda().eval()
        if float:
            G = G.float()

    if refresh:
        with torch.no_grad():
            old_G = G
            G = networks.Generator(*old_G.init_args, **old_G.init_kwargs).cuda()
            misc.copy_params_and_buffers(old_G, G, require_all=True)
            for param in G.parameters():
                param.requires_grad = False
    
    # FIXED: This must be outside the 'if refresh' block to ensure 
    # it applies to the generator regardless of the refresh setting.
    G.synthesis.forward = hooked_synthesis_forward(G.synthesis)
    return G

def hooked_synthesis_forward(synthesis_module):
    """
    Captures intermediate block activations during the forward pass.
    Activations are stored in synthesis_module.activations for loss calculation.
    """
    old_forward = synthesis_module.forward
    
    def new_forward(ws, **kwargs):
        synthesis_module.activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                # Standard StyleGAN2-ADA output is usually (img, features) or just img
                if isinstance(output, tuple):
                    # We take the first element which is the feature map
                    synthesis_module.activations[name] = output[0]
                else:
                    synthesis_module.activations[name] = output
            return hook

        handles = []
        # Target blocks that influence pFID/LPIPS balance
        # FIXED: Ensure we are looking at the right submodule level
        for name, module in synthesis_module.named_children():
            # StyleGAN2 blocks are usually named like 'b64', 'b128', etc.
            if any(res in name for res in ['64', '128', '256']):
                handles.append(module.register_forward_hook(hook_fn(name)))
        
        try:
            out = old_forward(ws, **kwargs)
        finally:
            # Crucial for Kaggle: remove hooks to prevent memory leaks
            for h in handles:
                h.remove()
                
        return out
    
    return new_forward
def open_image(path: str, resolution: int):
    image = TF.to_tensor(Image.open(path)).cuda().unsqueeze(0)[:, :3]
    image = TF.center_crop(image, min(image.shape[2:]))
    return F.interpolate(image, resolution, mode="area")


def resize_for_logging(x: torch.Tensor, resolution: int) -> torch.Tensor:
    return F.interpolate(
        x,
        size=(resolution, resolution),
        mode="nearest" if x.shape[-1] <= resolution else "area",
    )


@contextlib.contextmanager
def directory(dir_path: str) -> None: 
    "Context manager for entering a directory, while automatically creating it if it does not exist."
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cwd = os.getcwd()
    os.chdir(dir_path)
    yield
    os.chdir(cwd)
