
def get_user_prompt_completion(observation: str, memory: str, action_space: str):
    return f"""
[User]
Hello! Who are you?
[Assistant]
I am a helpful assistant.
<|endoftext|>
[User]
{get_user_prompt(observation, memory, action_space)}
[Assistant]
--- Thought ---
"""


def get_user_prompt(observation: str, memory: str, action_space: str):
    return f'''
You are an autonomous explorer. 
Your goal is to explore the environment interestingly.

## Action space
{action_space}

## Current Observation
{observation}

## Memory
{memory}

## Requirements

You must respond with the following structure:
--- Thought ---
(Reasoning why you choose this action.)
--- Action ---
(Action Should be one action of the action space. )

Decide the next move.
'''
