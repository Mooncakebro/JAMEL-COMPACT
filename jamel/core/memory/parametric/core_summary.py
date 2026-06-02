from typing import TYPE_CHECKING, List, Tuple

from jamel.core.env.web.utils import StepHistory
from jamel.core.memory.parametric.prompts import get_step_memory_summary_prompt_gemini
from jamel.log import log_utils
from jamel.models.service.base import ModelInterface
from jamel.utils.general_utils import make_assistant_serialized, summarize_str

if TYPE_CHECKING:
    from jamel.core.policy.agent import PolicyAgent

logger = log_utils.get_logger(__name__)


def add_memory_summaries_to_step_histories(model: ModelInterface, step_histories: List[StepHistory]):
    '''
    Synthesize ONLY the new experience summary for each step based on
    (current observation, action, next observation). This does NOT integrate
    with any prior memory content; callers can concatenate externally.
    '''
    step_update_list: List[Tuple[StepHistory, str]] = []
    for step_idx, step_history in enumerate(step_histories[:-1]):
        current_observation = step_history.before_observation
        current_action = step_history.parsed_content['action']
        current_memory = step_history.memory_content

        next_step_history = step_histories[step_idx + 1]
        next_observation = next_step_history.before_observation

        step_memory_prompt = get_step_memory_summary_prompt_gemini(
            current_observation=current_observation,
            current_action=current_action,
            next_observation=next_observation,
        )
        logger.info(f"try to get memory summary: \n{summarize_str(500)(step_memory_prompt)}")
        memory_summary = model.get_chat_response([
            make_assistant_serialized(step_memory_prompt)
        ])['choices'][0]['message']['content']
        logger.info(f"memory summary: \n{summarize_str(500)(memory_summary)}")
        step_update_list.append((step_history, current_memory, memory_summary))
    for step_history, current_memory, memory_summary in step_update_list:
        if current_memory:
            step_history.memory_content = f"{current_memory}\n{memory_summary}"
        else:
            step_history.memory_content = memory_summary


class ParametricMemorySummary:
    '''
    Construct parametric memory data by summarizing only the new experience
    at each step (observation, action, next observation).
    '''
    def __init__(self, policy_agent: 'PolicyAgent', data_synthize_model=None):
        self.policy_agent = policy_agent
        self.data_synthize_model = data_synthize_model or policy_agent.brain.model

    def reset(self):
        pass

    def update_memory(self):
        add_memory_summaries_to_step_histories(self.data_synthize_model, step_histories=self.policy_agent.history)

    def get_memory(self):
        pass
