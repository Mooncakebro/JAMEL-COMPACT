"""
应用配置管理
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from pydantic import Field, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource

class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """从 config.yaml 文件加载配置"""
    
    def __init__(self, settings_cls: type[BaseSettings]):
        super().__init__(settings_cls)
        # 修复点：在初始化时就读取文件，不要等到 get_field_value
        self._yaml_data = self._read_files()
    def _read_files(self) -> Dict[str, Any]:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--config', default='configs/config.yaml', help='配置文件路径')
        args, _ = parser.parse_known_args()
        yaml_file = args.config
        if not os.path.exists(yaml_file):
            # 增加打印提示，避免路径错误时静默失败
            raise FileNotFoundError(f"YAML 配置文件未找到: {yaml_file}")
        encoding = self.settings_cls.model_config.get('env_file_encoding', 'utf-8')
        with open(yaml_file, 'r', encoding=encoding) as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}

    def get_field_value(self, field: FieldInfo, field_name: str) -> Tuple[Any, str, bool]:
        return self._yaml_data.get(field_name), field_name, False
    def prepare_field_value(self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool) -> Any:
        return value
    def __call__(self) -> Dict[str, Any]:
        # 修复点：Pydantic V2 标准的 __call__ 写法，过滤掉 None 值，让低优先级(env)可以接管
        d: Dict[str, Any] = {}
        for field_name, field in self.settings_cls.model_fields.items():
            field_value, field_key, value_is_complex = self.get_field_value(field, field_name)
            field_value = self.prepare_field_value(field_name, field, field_value, value_is_complex)
            if field_value is not None:
                d[field_key] = field_value
        return d

# --- 2. 定义命令行参数加载源 ---
class CliSettingsSource(PydanticBaseSettingsSource):
    """
    从命令行参数加载配置
    """
    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> Tuple[Any, str, bool]:
        # 懒加载：只解析一次参数
        if not hasattr(self, '_cli_data'):
            self._cli_data = self._parse_args()
        
        field_value = self._cli_data.get(field_name)
        return field_value, field_name, False
    def prepare_field_value(
        self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool
    ) -> Any:
        return value
    def _parse_args(self) -> Dict[str, Any]:
        parser = argparse.ArgumentParser(description="Application Settings", argument_default=argparse.SUPPRESS, allow_abbrev=False)
        for name, field in self.settings_cls.model_fields.items():
            arg_name = f"--{name}"
            # arg_name = f"--{name.replace('_', '-')}"
            
            # 特殊处理 bool 类型，使其可以通过 --headless-mode 开启，无需传参数
            if field.annotation is bool or field.annotation is Optional[bool]:
                 parser.add_argument(
                    arg_name,
                    dest=name,
                    action="store_true", # 只要出现这个 flag 就为 True
                    help=field.description,
                )
                 # 也可以添加一个 --no-xxx 来设为 False，这里简化处理
            else:
                parser.add_argument(
                    arg_name,
                    dest=name,
                    help=field.description,
                )
        args, _ = parser.parse_known_args()
        return vars(args)
    def __call__(self) -> Dict[str, Any]:
        self._cli_data = self._parse_args()
        return self._cli_data

# --- 3. 配置类 ---
class Settings(BaseSettings):
    """应用配置类"""
    # --- 字段定义 (保持不变) ---
    exp_name: str = Field(default="main", description="本次实验的名称")
    # API 配置
    model_api_host: str = Field(default="0.0.0.0", description="API 服务器主机 (model service)")
    model_api_port: int = Field(default=8888, description="API 服务器端口 (model service)")
    jinja_template_path: Optional[str] = Field(default=None, description="模型推理使用的自定义 jinja 模板")
    model_api_key: str = Field(default="test", description="API 服务器 Key (model service)")
    inference_kwargs: dict = Field(default_factory=dict, description="推理超参数")
    # 模型配置
    model_type: Optional[str] = Field(default=None, description="模型类型")
    # 核心配置
    memory_type: str = Field(default='window', description="Memory Type")
    brain_type: str = Field(default='naive', description="Brain Type")
    # OpenAI 配置
    model_name: Optional[str] = Field(default=None, description="Model Name")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API Key")
    openai_base_url: Optional[str] = Field(default=None, description="OpenAI Base URL")
    # Web 自动化配置
    headless_mode: bool = Field(default=False, description="无头浏览器模式")
    record_coverage: bool = Field(default=False, description="使用 V8 引擎计算覆盖率")
    browser_timeout: int = Field(default=30000, description="浏览器超时时间（毫秒）")
    target_urls: list[str] = Field(default_factory=list, description="待探索的 URL 列表")
    url_parallelism: int = Field(default=1, description="不同 URL 之间的并行数")
    trajectory_parallelism_per_url: int = Field(default=1, description="同一 URL 下并行 trajectory 数")
    curriculum_stage_iterations: int = Field(default=1, description="每个 curriculum stage 内包含的 iteration 数", ge=1)
    max_steps_per_trajectory: int = Field(default=50, description="最大步数")
    default_env: str = Field(default="CartPole-v1", description="默认环境")
    learning_rate: float = Field(default=0.001, description="学习率")
    episodes: int = Field(default=1000, description="训练回合数")
    # 训练配置，兼容 swift/hf
    training_api_type: Optional[str] = Field(default=None, description="训练使用的后端")
    stage_training_args: Optional[dict] = Field(default=None, description="每个阶段指定的特定训练参数")
    start_iteration_step: Optional[int] = Field(default=0, description="初始的迭代步数.")
    model: Optional[str] = Field(default=None, description="初始模型")
    max_version_limit: int = Field(default=3, description='允许保存的模型版本数量最大值', ge=2)
    dataset_num_proc: int = Field(default=16, description="number of data processor")
    output_base_dir: Optional[str] = Field(default=None, description="模型保存的根目录")
    update_iterations: int = Field(default=10, description="模型迭代轮数")
    explore_num_per_iteration: int = Field(default=10, description="每次迭代的探索轮数")
    least_sample_num_per_iteration: int = Field(default=10, description="每次更新最少的样本训练数量")
    data_processors: dict = Field(default_factory=dict, description='Data Processor Config')
    filter_method: str = Field(default='top_k', description="数据筛选方案")
    # RL 配置
    # 日志配置
    log_level: str = Field(default="INFO", description="日志级别")
    log_file: Optional[str] = Field(default=None, description="日志文件路径")
    # 历史记录配置
    history_data_base_dir: str = Field(default="exploration_data", description="历史记录数据目录")
    # --- 配置部分 ---
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            CliSettingsSource(settings_cls),  # CLI 优先级最高
            # env_settings, # 环境变量
            dotenv_settings, # .env 文件
            YamlConfigSettingsSource(settings_cls), # YAML 其次
        )

    @field_validator("target_urls", mode="before")
    @classmethod
    def _parse_target_urls(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                parsed = json.loads(stripped)
                if not isinstance(parsed, list):
                    raise ValueError("target_urls JSON must decode to a list.")
                return [str(item).strip() for item in parsed if str(item).strip()]
            normalized = stripped.replace("\n", ",")
            return [item.strip() for item in normalized.split(",") if item.strip()]
        return [str(value).strip()]


# 全局配置实例
# 这里使用了 functools.lru_cache 类似的单例模式逻辑，但在 Python 模块级别直接实例化也是安全的
settings = Settings()

def get_settings() -> Settings:
    """获取配置实例"""
    return settings

if __name__ == "__main__":
    # 测试代码：打印当前配置
    import json
    print(json.dumps(settings.model_dump(), indent=2, default=str))
