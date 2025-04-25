import copy
import torch
import wandb
import random
import argparse
import numpy as np
import torch.nn as nn
from itertools import chain
from peft import LoraConfig
from L3LoraL3 import L3LoraL3
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, TrainingArguments, Trainer
import argparse

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def get_training_args():
    parser = argparse.ArgumentParser(
        description="Training script arguments for LLM fine-tuning with memory and PEFT support."
    )

    # === Project and Output Configuration ===
    parser.add_argument("--project_name", type=str, default="ACL SRW", 
                        help="Name of the project for logging and tracking.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save the model checkpoints and outputs.")
    parser.add_argument("--logging_dir", type=str, required=True, 
                        help="Directory for logging training metrics.")
    parser.add_argument("--save_strategy", type=str, choices=["steps", "epoch"], default="steps",
                        help="Checkpoint save strategy: 'steps' to save every `save_steps`, or 'epoch' to save at the end of each epoch.")
    parser.add_argument("--save_steps", type=int, default=300, 
                        help="Number of steps between saving checkpoints (used if save_strategy='steps').")
    parser.add_argument("--save_total_limit", type=int, default=1,
                        help="Maximum number of checkpoints to keep. Older ones will be deleted. Set to None to keep all.")

    # === Evaluation Configuration ===
    parser.add_argument("--eval_strategy", type=str, choices=["steps", "epoch", "no"], default="steps",
                        help="Evaluation strategy: 'steps', 'epoch', or 'no' for skipping evaluation.")
    parser.add_argument("--eval_steps", type=int, default=50, 
                        help="Number of steps between evaluations (used if eval_strategy='steps').")
    parser.add_argument("--eval_accumulation_steps", type=int, default=1, 
                        help="Number of accumulation steps for evaluation to reduce memory usage.")

    # === Model Configuration ===
    parser.add_argument("--model_name", type=str, default=None, 
                        help="Pretrained model name or path.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                        help="Path to a checkpoint to resume training from.")

    # === LoRA / PEFT Configuration ===
    parser.add_argument("--use_peft", action="store_true", default=False, 
                        help="Enable PEFT (e.g., LoRA) for parameter-efficient fine-tuning.")
    parser.add_argument("--lora_r", type=int, default=64, 
                        help="LoRA rank: dimensionality of the low-rank matrices.")
    parser.add_argument("--lora_alpha", type=int, default=32, 
                        help="LoRA alpha scaling factor.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, 
                        help="Dropout probability for LoRA layers.")

    # === Training Configuration ===
    parser.add_argument("--deepspeed_config", type=str, required=True, 
                        help="Path to DeepSpeed configuration JSON file.")
    parser.add_argument("--max_steps", type=int, default=10000, 
                        help="Total number of training steps.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, 
                        help="Initial learning rate.")
    parser.add_argument("--warmup_steps", type=int, default=300, 
                        help="Number of warmup steps before linear decay.")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup", 
                        help="Type of learning rate scheduler.")

    parser.add_argument("--per_device_train_batch_size", type=int, default=4, 
                        help="Training batch size per device.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, 
                        help="Evaluation batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="Number of gradient accumulation steps before optimizer step.")
    parser.add_argument("--logging_steps", type=int, default=20, 
                        help="Number of steps between logging metrics.")

    # === Input Sequence and Memory ===
    parser.add_argument("--max_length", type=int, default=200, 
                        help="Maximum input sequence length.")
    parser.add_argument("--num_mem", type=int, default=4, 
                        help="Number of memory slots used in memory-augmented transformer.")

    # === Dataset Configuration ===
    parser.add_argument("--dataset_name", type=str, default=None, 
                        help="Name of the dataset to use (from HuggingFace Datasets).")

    return parser.parse_args()

def setup_seed(seed: int, deterministic=False):
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # memory will be large when setting deterministic to True
    if torch.backends.cudnn.is_available() and deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 
        torch.use_deterministic_algorithms(True)


if __name__ == "__main__":
    args = get_training_args()
    setup_seed(42, True)
    device = torch.device(f"cuda")

    # download the dataset from huggingface
    raw_datasets = load_dataset(args.dataset_name)
    # train_indices = random.sample(range(len(raw_datasets['train'])), 1000)
    
    raw_datasets = DatasetDict({
        'train': raw_datasets['train'].select(range(100)),
        'validation': raw_datasets['validation'],
        'test': raw_datasets['test'],
    })

    # create the training and evaluation datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[-1]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], add_special_tokens=False, return_attention_mask=False)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    def group_texts(examples, max_length):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder
        total_length = (total_length // max_length) * max_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()
        }

        result["labels"] = [chunk + [128001] for chunk in result["input_ids"]]

        return result

    train_dataset = tokenized_datasets["train"].map(lambda x: group_texts(x, args.max_length),
                                                    batched=True, desc=f"Grouping train in chunks of {args.max_length}")

    valid_dataset = tokenized_datasets["validation"].map(lambda x: group_texts(x, args.max_length),
                                                    batched=True, desc=f"Grouping validation in chunks of {args.max_length}")                                               
    
    valid_dataset = valid_dataset.shuffle(seed=42).select(range(100))
    
    # lora configurations
    if args.use_peft:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

    wandb.init(project=args.project_name, name='500x_lora1_40k')

    # ====================
    # compression model
    # ====================
    model = L3LoraL3(
        model_name=args.model_name,
        max_length=args.max_length,
        use_peft=args.use_peft,
        lora_config=lora_config if args.use_peft else None,
        num_mem=args.num_mem,
        device=device
    )
    print("Number of trainable parameters in the model: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Model is on CUDA device:", torch.cuda.current_device())
    print("model.config: ", model.config)

    # ====================
    # Training
    # ====================
    # give the detailed information for the error
    torch.autograd.set_detect_anomaly(True)

    # training parameters
    training_args = TrainingArguments(
        output_dir=args.output_dir,         
        overwrite_output_dir=False,
        max_steps = args.max_steps,              
        per_device_train_batch_size=args.per_device_train_batch_size,   
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,      
        eval_strategy=args.eval_strategy,    
        eval_steps=args.eval_steps, 
        eval_accumulation_steps=args.eval_accumulation_steps,
        logging_dir=args.logging_dir,    
        logging_steps=args.logging_steps,
        deepspeed=args.deepspeed_config,
        learning_rate=args.learning_rate,
        save_total_limit=args.save_total_limit,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    # If the resume path is not none
    # continue from the provided checkpoint
    if args.resume_from_checkpoint == None:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=arg.resume_from_checkpoint)
    
    evaluation_results = trainer.evaluate()
    print("evaluation_results: ", evaluation_results)


