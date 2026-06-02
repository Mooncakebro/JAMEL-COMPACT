from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from verl.utils.import_utils import load_extern_type


def has_custom_model(custom_cfg: DictConfig | dict | None) -> bool:
    if custom_cfg is None:
        return False
    if isinstance(custom_cfg, DictConfig):
        return bool(custom_cfg.get("path", None))
    return bool(custom_cfg.get("path"))


def instantiate_custom_model(
    *,
    custom_cfg: DictConfig | dict,
    pretrained_model_name_or_path: str,
    model_config=None,
    torch_dtype=None,
    trust_remote_code: bool = False,
    extra_kwargs: dict[str, Any] | None = None,
):
    extra_kwargs = dict(extra_kwargs or {})
    custom_model_cls = load_extern_type(custom_cfg["path"], custom_cfg["name"])
    if hasattr(custom_model_cls, "from_pretrained"):
        return custom_model_cls.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            config=model_config,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **extra_kwargs,
        )
    return custom_model_cls(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        config=model_config,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        **extra_kwargs,
    )


def resolve_memory_augment_config(config_section: DictConfig | dict | None) -> dict[str, Any]:
    if config_section is None:
        return {}
    if isinstance(config_section, DictConfig):
        return OmegaConf.to_container(config_section, resolve=True)
    return dict(config_section)
