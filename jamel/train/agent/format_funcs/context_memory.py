
def format_web_explorer_example_w_context_memory(example: dict, get_user_prompt=None):
    if get_user_prompt is None:
        from jamel.core.policy.brains.naive.prompt import get_user_prompt
        get_user_prompt = get_user_prompt

    from jamel.core.env.web.action_space import get_action_space
    user_prompt = get_user_prompt(observation=example["before_observation_str"], memory=example['memory_content'], action_space = get_action_space())
    example['messages'] = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": example['raw_content']}
    ]
    return example
