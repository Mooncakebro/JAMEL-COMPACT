from functools import partial
from pathlib import Path
from jamel.config.settings import get_settings
from jamel.core.world_model.web.prompt import get_predict_prompt
from jamel.log import log_utils
from jamel.train.agent.main import run_training
from jamel.train.utils import prepare_processed_dataset
from jamel.train.world_model.data import format_web_world_model_example, world_model_processor

logger = log_utils.get_logger(__name__)

if __name__ == "__main__":
    settings = get_settings()
    saved_file_path = Path("exploration_data/archives/zhihu/histories/agent_history_20260203_015340.parquet")
    iteration_step = 0
    # 收集数据 & 训练逻辑...
    logger.info("Collecting data and training...")
    data_path = (Path(saved_file_path) / "..").resolve()
    processed_data_path = (data_path / ".." / "processed_data" / f"world_model_iteration_{iteration_step}.parquet").resolve()

    format_func = partial(format_web_world_model_example, get_user_prompt=get_predict_prompt)
    prepare_processed_dataset(data_path=str(data_path), processed_data_path=str(processed_data_path), processor=partial(world_model_processor, format_func=format_func), dataset_num_proc=settings.dataset_num_proc)
        
    output_base_dir = Path(settings.output_base_dir)
    output_dir = output_base_dir / f"world_model_iteration_{iteration_step}"
    
    output_model_dir = run_training(settings.training_api_type, data_path=str(processed_data_path), output_dir=output_dir)

# example: NPROC_PER_NODE=8 python jamel/train/world_model/main.py --config configs/config.yaml