'''
感觉太废了，ddp, deepspeed 之类的功能都要自己加，而且就是 transformers 套壳。感觉不如 CLI 版本好用。要用 API 版本，不如直接使用 transformers。
'''
import os
import sys
from datasets import load_dataset
from transformers import HfArgumentParser
from swift import get_model_processor, get_template
from swift.dataset import EncodePreprocessor
from swift.utils import get_logger, find_all_linears, get_model_parameter_info, plot_images, seed_everything
from peft import get_peft_model, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from jamel.train.agent.processor import format_web_explorer_example_w_context_memory
from jamel.train.agent.params import DataArguments, ModelArguments, TrainingArguments

logger = get_logger()

def train(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    seed_everything(training_args.seed)
    # Obtain the model and template, and add a trainable Lora layer on the model.
    model, tokenizer = get_model_processor(model_args.model_id_or_path)
    logger.info(f'model_info: {model.model_info}')
    template = get_template(tokenizer, max_length=training_args.max_seq_length) # 需要支持自定义聊天 Jinja 模板。
    template.set_mode('train')

    target_modules = find_all_linears(model)
    lora_config = LoraConfig(task_type='CAUSAL_LM', r=training_args.lora_rank, lora_alpha=training_args.lora_alpha,
                            target_modules=target_modules)
    model = get_peft_model(model, lora_config)
    logger.info(f'lora_config: {lora_config}')

    # Print model structure and trainable parameters.
    logger.info(f'model: {model}')
    model_parameter_info = get_model_parameter_info(model)
    logger.info(f'model_parameter_info: {model_parameter_info}')

    # Download and load the dataset, split it into a training set and a validation set,
    # and encode the text data into tokens.
    # TODO: 把这个放到外面去
    train_dataset = load_dataset(data_args.data_path, split='train', num_proc=data_args.dataset_num_proc)
    train_dataset = train_dataset.map(format_web_explorer_example_w_context_memory, num_proc=data_args.dataset_num_proc)


    logger.info(f'train_dataset: {train_dataset}')
    # logger.info(f'val_dataset: {val_dataset}')
    # logger.info(f'train_dataset[0]: {train_dataset[0]}')

    train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=data_args.dataset_num_proc)
    # val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc)

    # logger.info(f'encoded_train_dataset[0]: {train_dataset[0]}')

    # Print a sample
    template.print_inputs(train_dataset[0])
    # Get the trainer and start the training.
    model.enable_input_require_grads()  # Compatible with gradient checkpointing
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        template=template,
        train_dataset=train_dataset,
        # eval_dataset=val_dataset,
    )
    trainer.train()

    last_model_checkpoint = trainer.state.last_model_checkpoint
    logger.info(f'last_model_checkpoint: {last_model_checkpoint}')
    # Visualize the training loss.
    # You can also use the TensorBoard visualization interface during training by entering
    # `tensorboard --logdir '{output_dir}/runs'` at the command line.
    images_dir = os.path.join(training_args.output_dir, 'images')
    logger.info(f'images_dir: {images_dir}')
    plot_images(images_dir, training_args.logging_dir, ['train/loss'], 0.9)  # save images

if __name__ == "__main__":
    # from jamel.config.settings import get_settings
    # settings = get_settings()
    print(sys.argv)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    train(model_args, data_args, training_args)