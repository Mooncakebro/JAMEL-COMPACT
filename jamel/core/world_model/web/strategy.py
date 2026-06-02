from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import re
from concurrent.futures import ThreadPoolExecutor

from jamel.core.world_model.web.model import WorldModelBase
from jamel.log import log_utils
from jamel.models.service.base import ModelInterface

logger = log_utils.get_logger(__name__)

# ==========================================
# 1. 抽象策略接口
# ==========================================

class InputStrategy(ABC):
    @abstractmethod
    def process(self, observation: str, action: str, model: ModelInterface) -> str:
        """处理原始 Observation，返回给模型的 Prompt Context"""
        pass

class OutputStrategy(ABC):
    @abstractmethod
    def generate(self, processed_obs: str, action: str, raw_obs: str, model: ModelInterface, action_space_str: str) -> str:
        """根据处理后的输入生成预测结果，并还原为完整的新 Observation"""
        pass

# ==========================================
# 2. 输入策略具体实现 (Input Optimizations)
# ==========================================

class ViewPortInputStrategy(InputStrategy):
    """
    【输入优化：窗口裁剪】
    只保留 Action 操作对象附近的 DOM 节点。
    """
    def __init__(self, window_size=20):
        self.window_size = window_size

    def process(self, observation: str, action: str, model: ModelInterface) -> str:
        # 1. 解析 Action 中的 bid (例如 click('84'))
        bid_match = re.search(r"['\"]?(\d+)['\"]?", action)
        target_bid = bid_match.group(1) if bid_match else None
        
        if not target_bid:
            return observation[-2000:] # Fallback: 无法定位则截取末尾

        lines = observation.split('\n')
        target_idx = -1
        for i, line in enumerate(lines):
            if f"[{target_bid}]" in line:
                target_idx = i
                break
        
        if target_idx == -1:
            return observation # Fallback

        # 2. 裁剪
        start = max(0, target_idx - self.window_size)
        end = min(len(lines), target_idx + self.window_size)
        cropped_content = "\n".join(lines[start:end])
        
        return f"...(clipped)...\n{cropped_content}\n...(clipped)..."

class CompressionInputStrategy(InputStrategy):
    """
    【输入优化：压缩】
    调用模型将 DOM Tree 转换为自然语言描述 (Summary)。
    对应你提到的 DynaWeb/WMA 的形式不受限文本描述。
    """
    def process(self, observation: str, action: str, model: ModelInterface) -> str:
        prompt = (
            "Compress the following accessibility tree into a concise textual description "
            "that captures the key interactive elements and current state.\n"
            f"Tree:\n{observation[:8000]}..." # 限制一下防止此处就爆
        )
        summary = model.get_chat_response(messages=[{"role": "user", "content": prompt}])
        logger.info(f"Input Compressed: {summary}")
        return summary

class RawInputStrategy(InputStrategy):
    """不优化，直接使用"""
    def process(self, observation: str, action: str, model: ModelInterface) -> str:
        return observation
# 假设这是你原本的 prompt 引用
from .prompt import get_predict_prompt

# --- Input Strategy: 原样输入 ---
class RawInputStrategy(InputStrategy):
    """
    【基线输入策略】
    不做裁剪，不做压缩，保留完整的 Observation 文本。
    缺点：Token 消耗大，容易超出 Context Window。
    优点：保留了页面的全局信息，模型不会因为视野受限而产生幻觉。
    """
    def process(self, observation: str, action: str, model: ModelInterface) -> str:
        # 直接透传原始 observation
        return observation

# ==========================================
# 3. 输出策略具体实现 (Output Optimizations)
# ==========================================

class ChunkingOutputStrategy(OutputStrategy):
    """
    【输出优化：分块】
    先规划块，再并行生成，最后拼接。
    """
    def generate(self, processed_obs: str, action: str, raw_obs: str, model: ModelInterface, action_space_str: str) -> str:
        # Step 1: Planning
        plan_prompt = f"Given context:\n{processed_obs}\nAction: {action}\nIdentify 3-4 logical sections of the NEW page to generate. Return list only."
        plan_res = model.get_chat_response(messages=[{"role": "user", "content": plan_prompt}])
        # Mock parsing logic
        chunks = ["Header", "Main Content", "Footer"] 
        
        # Step 2: Parallel Generation
        results = {}
        def _gen_chunk(chunk_name):
            p = f"Predict the accessibility tree for section '{chunk_name}' after action {action}."
            return model.get_chat_response(messages=[
                {"role": "system", "content": f"Context: {processed_obs}"},
                {"role": "user", "content": p}
            ])

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(_gen_chunk, c): c for c in chunks}
            for f in futures:
                results[futures[f]] = f.result()
        
        return "\n".join([results[c] for c in chunks])

class DiffOutputStrategy(OutputStrategy):
    """
    【输出优化：差分】
    模型仅输出 Patch 指令，代码负责应用到原始 raw_obs 上。
    """
    def generate(self, processed_obs: str, action: str, raw_obs: str, model: ModelInterface, action_space_str: str) -> str:
        diff_prompt = (
            f"Observation Context: {processed_obs}\nAction: {action}\n"
            "Predict the changes. Output format:\n"
            "REMOVE <bid>\nREPLACE <bid> <new_content>\nADD <parent_bid> <content>"
        )
        diff_text = model.get_chat_response(messages=[{"role": "user", "content": diff_prompt}])
        
        # Apply Patch Logic (简化版)
        # 注意：这里必须用到 raw_obs，因为 Patch 是基于完整原页面的
        new_obs = raw_obs
        for line in diff_text.split('\n'):
            if line.startswith("REMOVE"):
                bid = line.split()[-1]
                # 极其简化的伪代码：在 new_obs 中删除含 bid 的行
                new_obs = "\n".join([l for l in new_obs.split('\n') if f"[{bid}]" not in l])
            # ... 处理 REPLACE / ADD ...
            
        return new_obs

class CompressionOutputStrategy(OutputStrategy):
    """
    【输出优化：压缩】
    模型直接输出自然语言描述，不再还原回 DOM Tree。
    """
    def generate(self, processed_obs: str, action: str, raw_obs: str, model: ModelInterface, action_space_str: str) -> str:
        prompt = (
            f"Current State: {processed_obs}\nAction: {action}\n"
            "Predict the next state as a concise text description. Do NOT output code."
        )
        description = model.get_chat_response(messages=[{"role": "user", "content": prompt}])
        return description

# --- Output Strategy: 完整预测 ---
class RawOutputStrategy(OutputStrategy):
    """
    【基线输出策略】
    预测完整的 Accessibility Tree / HTML。
    缺点：生成速度极慢（Latency 高），长文本生成的稳定性差（容易截断或循环）。
    优点：格式最标准，最容易与真实环境（Ground Truth）做 Diff 对比计算 Metric。
    """
    def generate(self, processed_obs: str, action: str, raw_obs: str, model: ModelInterface, action_space_str: str) -> str:
        # 1. 使用标准的 Prompt 模板构造输入
        # 注意：这里的 processed_obs 就是完整的 raw_obs
        prompt = get_predict_prompt(
            observation=processed_obs, 
            action=action, 
            action_space=action_space_str
        )

        # 2. 调用模型生成完整页面
        # 这里通常需要设置较大的 max_tokens，防止生成的 HTML 被截断
        response = model.get_chat_response(
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response
# ==========================================
# 4. 组合式 World Model
# ==========================================

class ModularWorldModel(WorldModelBase):
    def __init__(
        self,
        model: ModelInterface,
        get_action_space,
        input_strategy: InputStrategy,
        output_strategy: OutputStrategy
    ):
        self.model = model
        self.get_action_space = get_action_space
        self.input_strategy = input_strategy
        self.output_strategy = output_strategy

    def predict(self, observation, action):
        # 1. 输入处理 (Input Optimization)
        # processed_obs 可能是裁剪过的 DOM，也可能是压缩后的 Summary
        processed_obs = self.input_strategy.process(observation, action, self.model)
        logger.info(f"Input processed via {type(self.input_strategy).__name__}")

        # 2. 输出生成 (Output Optimization)
        # 注意：Output Strategy 可能需要 raw_obs (比如 Diff 模式需要做 Patch)
        prediction = self.output_strategy.generate(
            processed_obs=processed_obs,
            action=action,
            raw_obs=observation,
            model=self.model,
            action_space_str=self.get_action_space()
        )
        logger.info(f"Output generated via {type(self.output_strategy).__name__}")
        
        return prediction
