from pathlib import Path
import shutil
from jamel.log import log_utils

logger = log_utils.get_logger(__name__)


def prepare_processed_dataset(data_path: str | Path, processed_data_path: str | Path, processor, dataset_num_proc=1):
    data_path = Path(data_path)
    processed_data_path = Path(processed_data_path)
    if processed_data_path.exists():
        logger.info(f"try to remove processed data! path: {processed_data_path}")
        if processed_data_path.is_dir():
            shutil.rmtree(processed_data_path)
        else: # is file
            processed_data_path.unlink()

    data_path_str = str(data_path.resolve())
    logger.info(f"try to load data! path: {data_path_str}")
    from datasets import load_dataset
    train_dataset = load_dataset(data_path_str, split='train', num_proc=dataset_num_proc)
    logger.info("trying to process data", data_len=len(train_dataset))
    train_dataset = processor(train_dataset)
    logger.info("data processed!", data_len=len(train_dataset))

    train_dataset.to_parquet(str(processed_data_path.resolve()))
    logger.info(f"processed data saved! path: {processed_data_path}")