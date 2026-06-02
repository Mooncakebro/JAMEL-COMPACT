from dataclasses import dataclass
from typing import Any

from browsergym.utils.obs import flatten_axtree_to_str


@dataclass
class _WebObservation:
    last_action: str
    last_action_error: str
    open_pages_urls: tuple[str, ...]
    open_pages_titles: tuple[str, ...]
    active_page_index: Any
    axtree_txt: str

    @classmethod
    def from_gym_obs(cls, obs: dict) -> "_WebObservation":
        return cls(
            last_action=obs.get("last_action", ""),
            last_action_error=obs.get("last_action_error", ""),
            open_pages_urls=tuple(obs.get("open_pages_urls", ())),
            open_pages_titles=tuple(obs.get("open_pages_titles", ())),
            active_page_index=obs.get("active_page_index"),
            axtree_txt=flatten_axtree_to_str(obs["axtree_object"]),
        )

    @property
    def last_action_result(self) -> str:
        return self.last_action_error or "Success."


def get_observation(obs: dict | None) -> str:
    if not isinstance(obs, dict):
        return "Observation unavailable."

    web_observation = _WebObservation.from_gym_obs(obs)
    return f"""
Last Action: {web_observation.last_action}

Last Action Result: {web_observation.last_action_result}

Current open pages URLs:
{web_observation.open_pages_urls}

Current open pages titles:
{web_observation.open_pages_titles}

Current active page index:
{web_observation.active_page_index}

Current Observation:
{web_observation.axtree_txt}
""".strip()


def get_screenshot(obs: dict | None):
    if not isinstance(obs, dict):
        return None
    for key in ("screenshot", "image", "rgb"):
        image = obs.get(key)
        if image is not None:
            return image
    return None
