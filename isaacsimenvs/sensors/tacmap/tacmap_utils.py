import torch


@torch.jit.script
def deform_quantize(deform: torch.Tensor) -> torch.Tensor:
    deform *= 1e3
    mask = deform < 0.5
    deform[mask] /= 5e-3
    deform[~mask] = (deform[~mask] - 0.5) / 3e-2 + 100
    deform[deform > 255] = 255
    deform[deform < 0] = 0
    return deform.to(torch.uint8)
