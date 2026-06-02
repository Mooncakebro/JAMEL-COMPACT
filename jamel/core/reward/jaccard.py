from typing import List

def jamel_reward_fn_jaccard(current_state, past_states: List[str]) -> float:
    """
    基于 Jaccard 相似度的新颖性奖励。
    适合状态较长且词序不那么重要的情况。
    """
    if len(past_states) < 1:
        return 1.0
            
    # 简单的分词预处理，转为集合
    def get_tokens(text):
        return set(str(text).split())
    current_tokens = get_tokens(current_state)
    
    if not current_tokens: # 空状态处理
        return 0.0
    max_similarity = 0.0
    
    for past_state in past_states:
        past_tokens = get_tokens(past_state)
        if not past_tokens:
            continue
            
        # Jaccard = 交集 / 并集
        intersection = len(current_tokens.intersection(past_tokens))
        union = len(current_tokens.union(past_tokens))
        
        sim = intersection / union if union > 0 else 0.0
        if sim > max_similarity:
            max_similarity = sim
            
    # 可以加一个非线性变换，让微小的差异更明显
    # 例如：reward = (1 - max_similarity) ** 2
    return 1.0 - max_similarity

def jamel_label_fn_jaccard(current_state, past_states: List[str], threshold: float=0.5):
    reward = jamel_reward_fn_jaccard(current_state, past_states)
    if reward > threshold:
        return 1
    else:
        return 0