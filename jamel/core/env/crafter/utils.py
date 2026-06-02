from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

@dataclass
class StepHistory:
    step: Any
    obs: dict
    info: dict
    observation_str: str
    llm_completion: str
    memory_content: Any
    action: Dict[str, str]
    result: Dict[str, Any]
    timestamp: Any
    # New fields for enhanced metrics
    player_pos: Optional[Tuple[int, int]] = None  # (x, y) coordinates

    def __post_init__(self):
        # Extract player_pos from info if not explicitly provided
        if self.player_pos is None and self.info:
            pos = self.info.get('player_pos')
            if pos is not None:
                if isinstance(pos, np.ndarray):
                    self.player_pos = tuple(pos.tolist())
                else:
                    self.player_pos = tuple(pos)

    def to_dict(self):
        '''
        只能包含可以序列化的数据。
        '''
        return {
            "step": self.step,
            "observation_str": self.observation_str,
            "memory_content": self.memory_content,
            "llm_completion": self.llm_completion,
            "action": self.action,
            "result": self.result,
            "timestamp": str(self.timestamp),
            "player_pos": list(self.player_pos) if self.player_pos else None,
        }
