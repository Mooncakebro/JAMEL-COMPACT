from abc import ABC, abstractmethod
from typing import Any
from jamel.log import log_utils
from jamel.models.service.base import ModelInterface
from .prompt import get_predict_prompt, get_free_style_observation_prompt

logger = log_utils.get_logger(__name__)

class WorldModelBase(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict(self, observation, action) -> Any:
        pass

class WorldModelFreeStyle(WorldModelBase):
    def __init__(
        self,
        model: ModelInterface,
        get_action_space
    ):
        self.model = model
        self.get_action_space = get_action_space

    def predict(self, observation, action):
        free_style_observation_prompt = get_free_style_observation_prompt(observation)
        free_style_obs = self.model.get_chat_response(
            messages=[
                {"role": "user", "content": free_style_observation_prompt}
            ]
        )
        logger.info(f"got free style observation: {free_style_obs}")
        predict_prompt = get_predict_prompt(observation=free_style_obs, action=action, action_space=self.get_action_space())

        raw_response = self.model.get_chat_response(
            messages=[
                {"role": "user", "content": predict_prompt}
            ]
        )
        response = self.parse_result(raw_response)
        logger.info(f"got predict response: {response}")
        return response

    def parse_result(self, raw_response):
        return raw_response
    

class WorldModelNaive(WorldModelBase):
    def __init__(
        self,
        model: ModelInterface,
        get_action_space
    ):
        self.model = model
        self.get_action_space = get_action_space

    def predict(self, observation, action):
        predict_prompt = get_predict_prompt(observation=observation, action=action, action_space=self.get_action_space())

        raw_response = self.model.get_chat_response(
            messages=[
                {"role": "user", "content": predict_prompt}
            ]
        )
        response = self.parse_result(raw_response)
        logger.info(f"got predict response: {response}")
        return response

    def parse_result(self, raw_response):
        return raw_response["choices"][0]["message"]["content"]