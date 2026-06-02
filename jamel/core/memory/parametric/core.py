import json
from typing import TYPE_CHECKING, List, Tuple
from jamel.core.env.web.utils import StepHistory
from jamel.core.memory.parametric.prompts import get_step_memory_prompt_gemini, get_trace_from_action_observation_pairs
from jamel.log import log_utils
from jamel.models.service.base import ModelInterface
from jamel.utils.general_utils import make_assistant_serialized, make_user, make_user_messages, make_user_serialized, summarize_str
if TYPE_CHECKING:
    from jamel.core.policy.agent import PolicyAgent

logger = log_utils.get_logger(__name__)

def add_memories_to_step_histories(model: ModelInterface, step_histories: List[StepHistory]):
    '''
    合成用于训练到参数里的记忆数据。需要思考每个 observation 的作用域。按理说，我们应该给每个当下的 observation 以及后续的 action 生成一个 prompt。简单起见，我们先预测接下来的1步动态，作为记忆。模型要把新来的 observation 整合进过去生成好的记忆中。observation 里面可能包含历史信息，但是这个也是由 observer 模块或者其他上层模块决定的。
    我们假设是每执行一个 step 就会调用一次 get memory prompt 来清洗数据。而且目前只看下一步的动态。
    '''
    step_update_list: List[Tuple[StepHistory, str]] = []
    for step_idx, step_history in enumerate(step_histories[-2:-1]): # 只看最后一步的 transition
        current_observation = step_history.before_observation
        current_memory = step_history.memory_content # 主要是 query 时生成的 memory content
        current_action = step_history.parsed_content['action']
        
        next_step_history = step_histories[step_idx + 1]
        next_observation = next_step_history.before_observation

        following_trace = get_trace_from_action_observation_pairs([(current_action, next_observation)])
        step_memory_prompt = get_step_memory_prompt_gemini(current_observation=current_observation, following_trace=following_trace, current_memory=current_memory)
        logger.info(f"try to get memory: \n{summarize_str(500)(step_memory_prompt)}")
        updated_memory = model.get_chat_response([
            make_user_serialized(step_memory_prompt),
            make_assistant_serialized('')
        ])['choices'][0]['message']['content']
        # 将更新的 memory 插入 step 中
        logger.info(f"updated memory: \n{summarize_str(500)(updated_memory)}")
        step_update_list.append((step_history, updated_memory))
    
    for step_history, updated_memory in step_update_list:
        step_history.memory_content = updated_memory # 这里有一个问题就是，生成的 SFT 数据集中会同时包含旧的 memory 和新的 memory，在训练的时候这两种 memory 很可能会冲突，应该怎么排除旧的 memory 呢。

class ParametricMemory:
    '''
    构造用于参数训练的记忆数据
    '''
    def __init__(self, policy_agent: 'PolicyAgent', data_synthize_model=None):
        self.policy_agent = policy_agent
        self.data_synthize_model = data_synthize_model or policy_agent.brain.model # 默认使用当前的模型来合成数据。实际上可以改为使用 base model 来合成数据。

        # 在这里订阅一些函数，用来更新需要保存的数据.

    def reset(self):
        pass

    def update_memory(self):
        add_memories_to_step_histories(self.data_synthize_model, step_histories=self.policy_agent.history)

    def get_memory(self):
        pass