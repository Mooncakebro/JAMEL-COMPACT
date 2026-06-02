from dataclasses import dataclass
from typing import Any

@dataclass
class Decision:
    parsed_content: dict
    memory_content: str
    raw_content: str # llm 补全的内容
    llm_response: Any # llm 的完整相应

    def to_json(self):
        llm_response = self.llm_response
        return {
            "action": self.action,
            "parsed_content": self.parsed_content,
            "memory_content": self.memory_content,
            "raw_content": self.raw_content,
            "llm_response": str(llm_response),
        }

    @property
    def action(self):
        return self.parsed_content['action']