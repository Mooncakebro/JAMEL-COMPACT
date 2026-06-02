from dataclasses import dataclass
from functools import partial
from typing import Protocol
import datasets
from jamel.core.memory.types import MemoryType
import random
from datasets import Dataset, concatenate_datasets

@dataclass(frozen=True)
class ProcessorStats:
    filtered_count: int
    processed_count: int

class Processor(Protocol):
    def __call__(self, dataset: Dataset, format_func, *args, **kwds):
        return super().__call__(*args, **kwds)

import random
from datasets import Dataset, concatenate_datasets

def threshold_processor(
    dataset: Dataset,
    format_func,
    least_sample_num: int = 0,
    threshold: float = 0.5,
    dataset_num_proc: int = 1,
    return_stats: bool = False,
):
    """
    根据 reward 阈值过滤数据集。如果过滤后的样本数量不足，则进行有放回的随机采样补齐。
    最后对数据集应用格式化函数 format_func。
    """
    
    # 1. 首先进行过滤
    filtered_dataset = dataset.filter(
        lambda x: x['reward'] > threshold, 
        num_proc=dataset_num_proc, 
        load_from_cache_file=False
    )
    
    # 2. 检查数量是否满足 least_sample_num
    current_num = len(filtered_dataset)
    
    if least_sample_num > 0 and current_num < least_sample_num:
        num_needed = least_sample_num - current_num
        
        # 情况 A: 过滤后还有样本，从这些好样本中随机重复采样来补齐 (Oversampling)
        if current_num > 0:
            # 生成随机索引，允许重复 (replacement=True)
            # random.choices 在 Python 3.6+ 可用，用于有放回采样
            random_indices = random.choices(range(current_num), k=num_needed)
            supplement_dataset = filtered_dataset.select(random_indices)
            
            # 合并原过滤数据集和补齐的数据集 (注意这里传入的是列表)
            filtered_dataset = concatenate_datasets([filtered_dataset, supplement_dataset])
            
        # 情况 B: 过滤后一个满足条件的样本都没有 (Fallback 回退机制)
        else:
            # 这里的处理方式是从**原数据集**中强行采样，避免返回空集导致后续报错
            print(f"警告：没有任何样本满足 reward > {threshold}。作为保底，将从原始数据集中采样。")
            original_num = len(dataset)
            if original_num == 0:
                raise ValueError("原始数据集为空，无法采样。")
            
            random_indices = random.choices(range(original_num), k=least_sample_num)
            filtered_dataset = dataset.select(random_indices)

    # 3. 如果传入了格式化函数，则应用它
    if format_func is not None:
        filtered_dataset = filtered_dataset.map(
            format_func,
            num_proc=dataset_num_proc,
            load_from_cache_file=False
        )
        
    # 4. 返回最终处理好的数据集 (修复了原代码直接 return dataset 的问题)
    if return_stats:
        stats = ProcessorStats(
            filtered_count=current_num,
            processed_count=len(filtered_dataset),
        )
        return filtered_dataset, stats
    return filtered_dataset

def top_k_processor(
    dataset: datasets.Dataset,
    format_func,
    least_sample_num: int = 0,
    top_k: int = 0,
    percent: float = 0.0,
    dataset_num_proc: int = 1,
    return_stats: bool = False,
):
    dataset_len = len(dataset)
    top_k = max(top_k, round(dataset_len * percent), least_sample_num)
    dataset = dataset.sort("reward", reverse=True, load_from_cache_file=False)
    dataset = dataset.select(range(top_k))
    dataset = dataset.map(format_func, num_proc=dataset_num_proc, load_from_cache_file=False)
    if return_stats:
        stats = ProcessorStats(
            filtered_count=min(dataset_len, top_k),
            processed_count=len(dataset),
        )
        return dataset, stats
    return dataset

def no_filter_processor(
    dataset: datasets.Dataset,
    format_func,
    dataset_num_proc: int = 1,
    return_stats: bool = False,
    **kwargs,
):
    dataset = dataset.map(format_func, num_proc=dataset_num_proc, load_from_cache_file=False)
    if return_stats:
        stats = ProcessorStats(
            filtered_count=len(dataset),
            processed_count=len(dataset),
        )
        return dataset, stats
    return dataset

def get_processor_func(filter_method, format_func, least_sample_num, **kwargs) -> Processor:
    match filter_method:
        case 'top_k':
            return partial(top_k_processor, format_func=format_func, least_sample_num=least_sample_num, **kwargs)
        case 'threshold':
            return partial(threshold_processor, format_func=format_func, least_sample_num=least_sample_num, **kwargs)
        case 'no_filter':
            return partial(no_filter_processor, format_func=format_func, least_sample_num=least_sample_num, **kwargs)
        case _:
            raise KeyError(f"{filter_method} is not a valid method!")
