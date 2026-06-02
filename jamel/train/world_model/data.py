import datasets

def format_web_world_model_example(example: dict, get_user_prompt=None):
    if get_user_prompt is None:
        from jamel.core.policy.brains.naive.prompt import get_user_prompt
        get_user_prompt = get_user_prompt

    from jamel.core.env.web.action_space import get_action_space
    user_prompt = get_user_prompt(observation=example["before_observation_str"], action=example['parsed_content']['action'], action_space=get_action_space())
    example['messages'] = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": example['after_observation_str']}
    ]
    return example

def world_model_processor(dataset: datasets.Dataset, format_func, dataset_num_proc=1):
    dataset = dataset.map(format_func, num_proc=dataset_num_proc, load_from_cache_file=False)
    return dataset
