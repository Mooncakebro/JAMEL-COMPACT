from typing import List, Tuple
from jamel.core.env.web.utils import StepHistory


def get_trace_from_action_observation_pairs(pairs: List[Tuple[str, str]]):
    trace_str = ''
    for pair_idx, (action, observation) in enumerate(pairs):
        trace_str += f'''
        Action {pair_idx + 1}: 
        {action}
        Obseravtion {pair_idx + 1}:
        {observation}
        '''
    return trace_str

def get_step_memory_prompt(current_observation, following_trace, current_memory):
    return f'''你正在整理一个 Agent 执行产生的轨迹数据，需要整理出关于当前 observation 的知识和经验，主要是关于：在当前的 observation 下，未来执行什么动作会发生什么？你需要把它们整理为自然语言的形式。具体而言，我们会给你提供 (1)当前的 observation; (2) 在当前状态下执行动作的真实轨迹; (3) 过去已整理好的，关于当前 observation 所积累的知识和经验。你需要做的事情是从当前的轨迹中提取经验，并且和过去的知识经验整合在一起。大致的格式是：当前处于什么样的 observation 下，如果我做什么事情，会发生什么；如果我做什么事情，会发生什么，...。既要包含过去已有的经验，也要包含当前状态下执行动作的真实轨迹。

现在，当前的 observation 是：
{current_observation}

接下来执行的真实 trace:
{following_trace}

在当前状态下，已经积累整理的知识经验：
{current_memory}

现在，开始你的整理。

'''

def get_step_memory_prompt_gemini(current_observation, following_trace, current_memory):
    return f"""You are a Cognitive Memory Synthesizer for an autonomous agent. Your core objective is to analyze the agent's execution trajectory and consolidate actionable knowledge about its environment. 

Specifically, you need to deduce the action-outcome relationships under the given observation to answer: "In this specific state, if I execute a certain action, what will be the result?"

### Inputs Provided:
1. **Current Observation**: The environmental state the agent is currently perceiving.
2. **Following Trace**: The actual sequence of actions executed and their corresponding outcomes starting from this observation.
3. **Current Memory**: The previously accumulated knowledge and heuristics regarding this specific observation.

### Task Instructions:
1. **Analyze the Trace**: Extract new cause-and-effect (action -> outcome) experiences from the `Following Trace`.
2. **Integrate and Update**: Merge these newly extracted insights with the `Current Memory`. If the new trace provides updated outcomes or contradicts past memory, prioritize the latest findings. Eliminate redundancies.
3. **Format Strictly**: Express the synthesized knowledge in clear, natural English using the following logical pattern:
   "Under the current observation (where [briefly summarize the key contextual features]), if I [Action A], then [Outcome A] will happen; if I [Action B], then [Outcome B] will happen..."

### Constraints:
- Ensure the final output is comprehensive: it MUST seamlessly encompass BOTH the valid past experiences from `Current Memory` AND the new insights from the `Following Trace`.
- Do NOT output conversational filler (e.g., "Here is the updated memory..."). Output ONLY the synthesized memory paragraph.

---
### Data

**Current Observation:**
{current_observation}

**Following Trace:**
{following_trace}

**Current Memory:**
{current_memory}

### Synthesized Memory:
"""

def get_step_memory_prompt_claude(current_observation, following_trace, current_memory):
    return f'''### Role
You are a Knowledge Synthesis Expert for Autonomous Agents. Your task is to analyze agent trajectories and extract actionable insights to update the agent's long-term memory.

### Context
An agent is navigating an environment. To improve its future decision-making, we need to document the causal relationships between its observations, actions, and the resulting outcomes.

### Input Data
1. **Current Observation**: The state the agent is currently perceiving.
   - Data: {current_observation}
   
2. **Execution Trace**: The actual sequence of actions taken and the outcomes observed following the current state.
   - Data: {following_trace}
   
3. **Existing Memory**: Previously synthesized knowledge regarding this specific observation.
   - Data: {current_memory}

### Task
Synthesize the "Execution Trace" with the "Existing Memory" to create a unified, natural language knowledge entry. 
Your output must follow this logic: "In [Observation X], performing [Action A] leads to [Outcome B]; however, performing [Action C] results in [Outcome D]."

### Requirements
- **Integration**: Seamlessly merge new insights from the trace into the existing memory without redundancy.
- **Causality**: Focus strictly on the causal link between actions and consequences under the given observation.
- **Clarity**: Use precise, professional natural language.
- **Format**: Provide a cohesive paragraph or a structured list of "If-Then" relationships.

### Output
Based on the data above, provide the updated memory synthesis:
'''

def get_step_memory_summary_prompt_gemini(current_observation, current_action, next_observation):
    return f"""You are a Memory Summarizer for an autonomous agent.

Your task is to produce a concise, factual summary of the NEW experience only, based on:
- Current Observation
- Action Taken
- Resulting Observation

### Requirements
- Focus strictly on the causal link between the action and the outcome under the given observation.
- Do NOT integrate or reference any prior memory.
- Output a single short paragraph (1-2 sentences), no bullets, no extra labels.

---
### Data

Current Observation:
{current_observation}

Action Taken:
{current_action}

Resulting Observation:
{next_observation}

### Summary:
"""
