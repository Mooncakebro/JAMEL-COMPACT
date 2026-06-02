def format_assistant_memory_message(example):
    return f'''--- Memory ---
{example['memory_content']}
'''
def format_web_explorer_example_w_parametric_memory(example: dict, get_user_prompt=None):
    '''
    参数化的 memory 格式化。
    '''
    if get_user_prompt is None:
        from jamel.core.policy.brains.parametric_memory.prompt import get_user_prompt
        get_user_prompt = get_user_prompt

    from jamel.core.env.web.action_space import get_action_space
    user_prompt = get_user_prompt(observation=example["before_observation_str"], action_space = get_action_space())
    example['messages'] = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": format_assistant_memory_message(example)}
    ] # 注意，在训练的时候 tokenizer 不能添加 eos。
    return example
    # 注意，参数化的 memory 不需要

def format_web_explorer_example_w_parametric_memory_policy(example: dict, get_user_prompt=None):
    """
    参数化 memory 的 policy 训练格式：
    - 输入：观测 + action space
    - 输出：模型原始决策（包含 Memory/Thought/Action）
    """
    if get_user_prompt is None:
        from jamel.core.policy.brains.parametric_memory.prompt import get_user_prompt
        get_user_prompt = get_user_prompt

    from jamel.core.env.web.action_space import get_action_space
    user_prompt = get_user_prompt(observation=example["before_observation_str"], action_space=get_action_space())
    example['messages'] = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": example['raw_content']}
    ]
    return example
