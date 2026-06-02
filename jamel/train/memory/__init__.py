__all__ = [
    "MemoryAugmentedCausalLM",
    "MemoryAugmentedValueModel",
    "MemoryTokenSFTDataset",
    "JAMELMemoryVLTokenSFTDataset",
    "OnlineHistoryMemoryBuilder",
]

_IMPORTS = {
    "MemoryAugmentedCausalLM": (".modeling", "MemoryAugmentedCausalLM"),
    "MemoryAugmentedValueModel": (".modeling", "MemoryAugmentedValueModel"),
    "MemoryTokenSFTDataset": (".sft_dataset", "MemoryTokenSFTDataset"),
    "JAMELMemoryVLTokenSFTDataset": (".jamel_sft_dataset", "JAMELMemoryVLTokenSFTDataset"),
    "OnlineHistoryMemoryBuilder": (".encoder", "OnlineHistoryMemoryBuilder"),
}


def __getattr__(name):
    if name not in _IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attr_name = _IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
