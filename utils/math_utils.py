import torch


def log(t, eps: float = 1e-12):
    r"""
    torch log with eps clampping
    """
    return torch.log(t.clamp(min=eps))


def l2normalization(input, p=2, dim=1, eps=1e-12):
    r"""
    Custom implementation of the 'torch.nn.functional.normalize'
    NOTE: the original F.normalize() raises error:
        'OpenCL error CL_INVALID_KERNEL_NAME for the kernel NormNumForwardContiguous.'
    """
    denom = torch.sqrt(torch.sum(input.pow(p), dim=dim).clamp_min(eps))
    denom = denom.unsqueeze(dim).expand_as(input)
    return input / denom
