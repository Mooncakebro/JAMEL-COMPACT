from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class BaseStepHistory:
    before_info: dict
    before_obs: dict
    step: Any
    reward: float