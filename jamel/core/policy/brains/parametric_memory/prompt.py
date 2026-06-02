def get_user_prompt(observation: str, action_space: str):
    return f'''You are an intelligent autonomous explorer. Your goal is to conduct a systematic and engaging exploration of the environment, uncovering hidden details and interesting patterns. Try your best to come up with novel actions and discover new things.

--- Valid Actions ---
{action_space}

--- Current Observation ---
{observation}

--- Instructions ---

You must respond strictly using the format below. Do not add any text outside these sections.

--- Memory ---
(Recall relevant experiences from memory. Based on the memory, you can propose some novel actions different from the past.)

--- Thought ---
(Think step-by-step. Analyze the current observation, recall your memory, and determine the most strategic next move.)

--- Action ---
(Choose exactly one action from the Valid Actions provided above.)

Now give your response based on all the requirements above.

'''
