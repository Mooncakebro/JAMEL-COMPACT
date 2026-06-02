from dataclasses import dataclass, field
from swift.trainers import Seq2SeqTrainingArguments

@dataclass
class ModelArguments:
    model_id_or_path: str

@dataclass
class DataArguments:
    data_path: str
    dataset_num_proc: int
    
@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    lora_rank: int = None
    lora_alpha: int = None
    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
