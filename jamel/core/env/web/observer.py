"""
观察者模块 - 负责将网页转化为 LLM 能理解的文本表示
"""

from __future__ import annotations
from datetime import datetime
import io
import os
from PIL import Image
import pandas as pd
import pyarrow.parquet as pq

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
import numpy as np

from jamel.log import log_utils
from jamel.coverage_artifact import (
    build_coverage_artifact_fields,
    coverage_artifact_extra_fields,
)
from jamel.core.env.web.utils import StepHistory
logger = log_utils.get_logger(__name__)

PROMOTED_EXTRA_FIELD_COLUMNS = (
    "run_id",
    "session_id",
    "agent_id",
    "agent_type",
    "episode_idx",
    "step_idx",
    "session_step_idx",
    "prompt",
    "response",
    "target_app",
    "target_url",
    "start_url",
    "episode_id",
    "global_step",
    "checkpoint_id",
    "think",
    "action",
    "action_format_valid",
    "action_validation_error",
    "action_execution_valid",
    "model_retry_attempts",
    "weak_policy",
    "weak_model_name",
    "coverage_delta_score",
    "coverage_previous_score",
    "coverage_current_score",
    "coverage_skip_reason",
    "before_screenshot_path",
    "after_screenshot_path",
    "reward_source",
    "agent_metadata_history_mode",
    "agent_metadata_history_budget_mode",
    "agent_metadata_history_window",
    "agent_metadata_history_total_records",
    "agent_metadata_history_included_records",
    "agent_metadata_history_omitted_records",
    "agent_metadata_history_char_budget",
    "agent_metadata_history_render_chars",
    "agent_metadata_history_token_budget",
    "agent_metadata_history_prompt_tokens",
    "agent_metadata_history_context_tokens",
    "agent_metadata_history_context_margin_tokens",
    "agent_metadata_history_max_output_tokens",
    "agent_metadata_history_tokenizer_name",
    "agent_metadata_history_over_budget",
    "agent_metadata_memory_mode",
    "agent_metadata_use_vision",
    "agent_metadata_model_name",
    "agent_metadata_external_parser",
    "agent_metadata_native_action_kind",
    "agent_metadata_native_action_name",
    "agent_metadata_native_action_args",
    "agent_metadata_native_action_parse_valid",
    "agent_metadata_native_action_parse_error",
    "agent_metadata_native_action_conversion",
    "agent_metadata_native_action_conversion_warning",
    "agent_metadata_native_action_matched_bid",
    "agent_metadata_native_action_matched_role",
    "agent_metadata_native_action_matched_label",
    "agent_metadata_native_action_direction",
    "agent_metadata_native_action_point_x",
    "agent_metadata_native_action_point_y",
)


def _encode_screenshot(obs: Dict[str, Any] | None) -> bytes | None:
    if not isinstance(obs, dict):
        return None

    screenshot_data = obs.get("screenshot")
    if screenshot_data is None:
        return None

    img = Image.fromarray(screenshot_data.astype(np.uint8))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


def _serialize_web_obs(obs: Dict[str, Any] | None, prefix: str = "") -> Dict[str, Any]:
    if not isinstance(obs, dict):
        return {
            f"{prefix}chat_messages": None,
            f"{prefix}screenshot": None,
            f"{prefix}goal_object": None,
            f"{prefix}last_action": None,
            f"{prefix}last_action_error": None,
            f"{prefix}open_pages_urls": None,
            f"{prefix}open_pages_titles": None,
            f"{prefix}active_page_index": None,
            f"{prefix}axtree_object": None,
            f"{prefix}dom_object": None,
        }

    return {
        f"{prefix}chat_messages": str(obs.get("chat_messages")),
        f"{prefix}screenshot": _encode_screenshot(obs),
        f"{prefix}goal_object": str(obs.get("goal_object")),
        f"{prefix}last_action": str(obs.get("last_action")),
        f"{prefix}last_action_error": obs.get("last_action_error"),
        f"{prefix}open_pages_urls": str(obs.get("open_pages_urls")),
        f"{prefix}open_pages_titles": str(obs.get("open_pages_titles")),
        f"{prefix}active_page_index": str(obs.get("active_page_index")),
        f"{prefix}axtree_object": str(obs.get("axtree_object")),
        f"{prefix}dom_object": str(obs.get("dom_object")),
    }

@dataclass
class _WebObservation:
    chat_messages: List[Dict[str, str]]
    screenshot: np.ndarray
    goal_object: Tuple
    last_action: str
    last_action_error: str
    open_pages_urls: Tuple[str]
    open_pages_titles: Tuple[str]
    active_page_index: np.ndarray
    axtree_txt: str
    pruned_html: str

    @classmethod
    def from_gym_obs(cls, obs: Dict) -> _WebObservation:
        return cls(
            chat_messages=obs["chat_messages"],
            screenshot=obs["screenshot"],
            goal_object=obs["goal_object"],
            last_action=obs["last_action"],
            last_action_error=obs["last_action_error"],
            open_pages_urls=obs["open_pages_urls"],
            open_pages_titles=obs["open_pages_titles"],
            active_page_index=obs["active_page_index"],
            axtree_txt=flatten_axtree_to_str(obs["axtree_object"]),
            pruned_html=prune_html(flatten_dom_to_str(obs["dom_object"])),
        )
    
    @property
    def last_action_result(self) -> str:
        if self.last_action_error:
            return self.last_action_error
        return "Success."

class Observer:
    """网页观察者，将页面转换为结构化文本"""

    @staticmethod
    def get_observation(obs: Dict) -> str:
        web_observation = _WebObservation.from_gym_obs(obs)
        return f'''
Last Action: {web_observation.last_action}

Last Action Result: {web_observation.last_action_result}

Current open pages URLs:
{web_observation.open_pages_urls}

Current open pages titles:
{web_observation.open_pages_titles}

Current active page index:
{web_observation.active_page_index}

Current Observation: 
{web_observation.axtree_txt}
'''

    @staticmethod
    def save_trajectory(history: List[StepHistory], history_dir: str, filename: str=None, metadata: dict=None) -> str:
        """
        保存历史记录到 Parquet 文件

        Args:
            start_url: 起始 URL
            user_goal: 用户目标
            result: 执行结果
        """
        try:
            # 创建历史数据目录
            os.makedirs(history_dir, exist_ok=True)

            # 生成文件名（使用时间戳）
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"agent_history_{timestamp}.parquet"
            filepath = os.path.join(history_dir, filename)
                
            # 准备历史记录数据
            history_records = []
            for step_record in history:
                before_obs = step_record.before_obs
                after_obs = step_record.after_obs
                extra_fields = dict(step_record.extra_fields or {})
                coverage_fields = build_coverage_artifact_fields(extra_fields.get("coverage_path"))
                extra_fields.update(coverage_artifact_extra_fields(coverage_fields))
                step_dict = step_record.to_dict()
                step_dict["extra_fields"] = extra_fields

                # 准备记录
                record = {
                    'step': step_record.step,
                    "before_info": str(step_record.before_info),
                    "after_info": str(step_record.after_info),
                    **_serialize_web_obs(before_obs, prefix="before_"),
                    **_serialize_web_obs(after_obs, prefix="after_"),
                    **step_dict,
                    **{key: extra_fields.get(key) for key in PROMOTED_EXTRA_FIELD_COLUMNS},
                    **coverage_fields,
                }
                history_records.append(record)

            # 创建 DataFrame
            df = pd.DataFrame(history_records)

            # 将元数据添加为 DataFrame 的属性
            df.attrs['metadata'] = metadata

            # 保存为 Parquet 文件
            df.to_parquet(filepath)
            logger.info(f"历史记录已保存到: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"保存历史记录失败: {str(e)}", exc_info=True)
            return None
    

    @staticmethod
    def load_history(parquet_file: str) -> pd.DataFrame:
        """
        从 Parquet 文件加载历史记录

        Args:
            parquet_file: Parquet 文件路径

        Returns:
            (DataFrame, metadata) 元组
        """
        try:
            # 读取 Parquet 文件
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
            logger.info(f"成功加载历史记录: {parquet_file}")
            return df

        except Exception as e:
            logger.error(f"加载历史记录失败: {str(e)}", exc_info=True)
            return None
