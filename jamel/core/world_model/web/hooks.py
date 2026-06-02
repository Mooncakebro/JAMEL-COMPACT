from jamel.core.policy.hooks import HookContext
from jamel.core.reward.jaccard import jamel_reward_fn_jaccard
from jamel.core.world_model.web.model import WorldModelBase
from jamel.log import log_utils


logger = log_utils.get_logger(__name__)

def hook_after_execute_action(ctx: HookContext, world_model: WorldModelBase, novelty_func=jamel_reward_fn_jaccard):
    '''
    在执行完成动作之后调用 world model 预测。计算 reward，并且保存数据用于训练。
    '''
    logger.info("call world model", world_model=world_model)
    
    observation = ctx.data.before_observation
    action = ctx.data.action
    new_observation = ctx.data.after_observation

    predicted_new_observation = world_model.predict(observation=observation, action=action)
    logger.info('predicted next observation: ', predicted_new_observation)
    novelty = novelty_func(predicted_new_observation, [new_observation])
    ctx.data.reward = novelty
    logger.info("calculated novelty!", novelty=novelty)
