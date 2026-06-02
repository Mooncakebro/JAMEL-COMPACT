
def get_predict_prompt(observation: str, action: str, action_space: str):
    return f'''
You are a web world model. Your goal is to predict the next observation of the web page, given the current observation and current action.

## Action space
{action_space}

## Current Observation
{observation}

## Current Action
{action}

Now, predict the next observation.
'''


def get_free_style_observation_prompt(observation: str):
    return f'''

## Current Observation
{observation}

Now, summarize the observation.
'''
