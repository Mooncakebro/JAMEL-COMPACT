# Copyright (c) ModelScope Contributors. All rights reserved.
from enum import Enum, auto
from functools import partial
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

import json

from jamel.log import log_utils

logger = log_utils.get_logger(__name__)

ROUTE_MAPPING: Dict[str, str] = {
    'swift': 'jamel.train.agent.swift.train_swift',
    'hf': 'jamel.train.agent.hf.train_hf',
}

def use_torchrun() -> bool:
    nproc_per_node = os.getenv('NPROC_PER_NODE')
    nnodes = os.getenv('NNODES')
    if nproc_per_node is None and nnodes is None:
        return False
    return True


def get_torchrun_args() -> Optional[List[str]]:
    if not use_torchrun():
        return
    torchrun_args = []
    for env_key in ['NPROC_PER_NODE', 'MASTER_PORT', 'NNODES', 'NODE_RANK', 'MASTER_ADDR']:
        env_val = os.getenv(env_key)
        if env_val is None:
            continue
        torchrun_args += [f'--{env_key.lower()}', env_val]
    return torchrun_args

def prepare_config_args(argv, api_args: Dict=None):
    for i in range(len(argv)):
        if argv[i] == '--config':
            if i + 1 >= len(argv):
                raise ValueError('The `--config` argument requires a yaml file path.')
            from omegaconf import OmegaConf, DictConfig, ListConfig
            config = OmegaConf.load(argv[i + 1])

            def parse_dict_config(cfg: DictConfig) -> Dict[str, Any]:
                result = {}
                for key, value in cfg.items():
                    if isinstance(value, DictConfig):
                        result[key] = json.dumps(OmegaConf.to_container(value))
                    elif isinstance(value, ListConfig):
                        result[key] = list(value)
                    else:
                        result[key] = value
                return result

            # Convert yaml to cmd line
            cfg = parse_dict_config(config)
            
            # add api args to cfg
            if api_args is not None:
                logger.info("add api args!", api_args=api_args)
                cfg.update(api_args)
            
            for key, value in cfg.items():
                argv.append(f'--{key}')
                if isinstance(value, list):
                    argv.extend(value)
                else:
                    argv.append(str(value))

            # Pop --config
            argv.pop(i)
            # Pop value of --config
            argv.pop(i)
            break

def _run_training_swift_api(method_name: str, route_mapping: Optional[Dict[str, str]] = None, is_megatron: bool = False, **api_args) -> None:
    route_mapping = route_mapping or ROUTE_MAPPING
    argv = sys.argv[1:]
    file_path = importlib.util.find_spec(route_mapping[method_name]).origin
    torchrun_args = get_torchrun_args()
    prepare_config_args(argv, api_args)
    python_cmd = sys.executable
    if torchrun_args is None or (not is_megatron and method_name not in {'swift', 'hf'}):
        args = [python_cmd, file_path, *argv]
    else:
        args = [python_cmd, '-m', 'torch.distributed.run', *torchrun_args, file_path, *argv]
    print(f"run sh: `{' '.join(args)}`", flush=True)
    result = subprocess.run(args)
    if result.returncode != 0:
        sys.exit(result.returncode)

def prepare_explorer_data(
    data_path: str | Path,
    processed_data_path: str | Path,
    processor,
    dataset_num_proc: int = 1,
    return_stats: bool = False,
):
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
    # train_dataset = train_dataset.map(processor, num_proc=dataset_num_proc, load_from_cache_file=False)
    logger.info("trying to process data", data_len=len(train_dataset))
    processed = processor(train_dataset, dataset_num_proc=dataset_num_proc, return_stats=return_stats)
    if return_stats:
        train_dataset, stats = processed
    else:
        train_dataset, stats = processed, None
    logger.info("data processed!", data_len=len(train_dataset))

    train_dataset.to_parquet(str(processed_data_path.resolve()))
    logger.info(f"processed data saved! path: {processed_data_path}")
    if return_stats:
        return stats

def _swift_sft(*args):
    full_args = ["swift", "sft", *args]
    print(f"start swift cli: `{' '.join(full_args)}`", flush=True)
    result = subprocess.run(full_args)
    if result.returncode != 0:
        sys.exit(result.returncode)
    return result

def _run_training_swift_cli(data_path: str | Path, output_dir: str | Path, **kwargs) -> str:
    '''
    返回值：结束时保存的参数路径。
    '''
    data_path = Path(data_path)
    output_dir = Path(output_dir)

    argv = sys.argv[1:] # 复制了一份 system args 放在这里，保证不会修改原始的 system args。
    prepare_config_args(argv)
    print(f"system args: {argv}")

    extra_args = []
    for key, value in kwargs.items():
        extra_args.append(f'--{key}')
        if isinstance(value, list):
            extra_args.extend(value)
        else:
            extra_args.append(str(value))
    
    _swift_sft(*argv, *extra_args, '--dataset', str(data_path.resolve()), '--output_dir', str(output_dir.resolve()), "--add_version", "false")

    checkpoints = list(output_dir.glob("checkpoint-*"))
    if checkpoints:
        last_model_checkpoint = max(checkpoints, key=lambda p: int(p.name.split('-')[-1]))
        return str(last_model_checkpoint)
    raise FileNotFoundError(f"Failed to find output model directory in {output_dir}!")

class TrainingAPIType(Enum):
    swift_api = auto()
    swift_cli = auto()

TRAINING_MAPPING: Dict[str, Any] = {
    TrainingAPIType.swift_api: _run_training_swift_api,
    TrainingAPIType.swift_cli: _run_training_swift_cli,
}

def run_training(training_api_type: str | TrainingAPIType, data_path, output_dir, **kwargs) -> str:
    return TRAINING_MAPPING[TrainingAPIType[training_api_type] if isinstance(training_api_type, str) else training_api_type](data_path=data_path, output_dir=output_dir, **kwargs)

if __name__ == '__main__':
    raise SystemExit(
        "This module is a library entry point. Use shell/run_qwen25vl_7b_sft.sh "
        "for the JAMEL MemoryAug SFT recipe."
    )
