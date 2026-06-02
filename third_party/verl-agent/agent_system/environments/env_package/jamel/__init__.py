from .projection import jamel_projection


def build_jamel_envs(*args, **kwargs):
    from .envs import build_jamel_envs as _build_jamel_envs

    return _build_jamel_envs(*args, **kwargs)
