from typing import Protocol

class FormatFunc(Protocol):
    def __call__(self, example: dict, get_user_prompt, *args, **kwds):
        return super().__call__(example, get_user_prompt, *args, **kwds)
