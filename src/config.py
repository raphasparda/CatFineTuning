"""
Configurações e constantes do projeto de Fine-Tuning.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class ModelConfig:
    """Configurações do modelo."""
    name: str = "meta-llama/Llama-3.2-3B-Instruct"
    quantization_enabled: bool = True
    quantization_bits: int = 4
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoRAConfig:
    """Configurações de LoRA."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class TrainingConfig:
    """Configurações de treinamento."""
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    fp16: bool = False
    bf16: bool = True
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    logging_steps: int = 10
    logging_first_step: bool = True
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_32bit"
    max_grad_norm: float = 0.3


@dataclass
class DatasetConfig:
    """Configurações do dataset."""
    path: str = "mlabonne/guanaco-llama2-1k"
    text_column: str = "text"
    instruction_column: str = "instruction"
    response_column: str = "response"
    max_seq_length: int = 1024
    train_split: float = 0.9
    seed: int = 42


@dataclass
class MLflowConfig:
    """Configurações de MLflow."""
    experiment_name: str = "llm-fine-tuning"
    tracking_uri: Optional[str] = None
    tags: dict = field(default_factory=lambda: {
        "project": "fine-tuning-pipeline",
        "framework": "transformers",
        "method": "qlora"
    })


@dataclass
class OutputConfig:
    """Configurações de output."""
    dir: str = "./outputs"
    model_name: str = "fine-tuned-model"
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None


@dataclass
class PipelineConfig:
    """Configuração completa do pipeline."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """Carrega configurações de um arquivo YAML."""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(
                name=config_dict.get('model', {}).get('name', 'meta-llama/Llama-3.2-3B-Instruct'),
                quantization_enabled=config_dict.get('model', {}).get('quantization', {}).get('enabled', True),
                quantization_bits=config_dict.get('model', {}).get('quantization', {}).get('bits', 4),
                bnb_4bit_compute_dtype=config_dict.get('model', {}).get('quantization', {}).get('bnb_4bit_compute_dtype', 'float16'),
                bnb_4bit_quant_type=config_dict.get('model', {}).get('quantization', {}).get('bnb_4bit_quant_type', 'nf4'),
                bnb_4bit_use_double_quant=config_dict.get('model', {}).get('quantization', {}).get('bnb_4bit_use_double_quant', True),
            ),
            lora=LoRAConfig(
                r=config_dict.get('lora', {}).get('r', 16),
                lora_alpha=config_dict.get('lora', {}).get('lora_alpha', 32),
                lora_dropout=config_dict.get('lora', {}).get('lora_dropout', 0.05),
                bias=config_dict.get('lora', {}).get('bias', 'none'),
                task_type=config_dict.get('lora', {}).get('task_type', 'CAUSAL_LM'),
                target_modules=config_dict.get('lora', {}).get('target_modules', [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]),
            ),
            training=TrainingConfig(
                num_train_epochs=config_dict.get('training', {}).get('num_train_epochs', 3),
                per_device_train_batch_size=config_dict.get('training', {}).get('per_device_train_batch_size', 4),
                per_device_eval_batch_size=config_dict.get('training', {}).get('per_device_eval_batch_size', 4),
                gradient_accumulation_steps=config_dict.get('training', {}).get('gradient_accumulation_steps', 4),
                learning_rate=config_dict.get('training', {}).get('learning_rate', 2e-4),
                weight_decay=config_dict.get('training', {}).get('weight_decay', 0.01),
                warmup_ratio=config_dict.get('training', {}).get('warmup_ratio', 0.03),
                lr_scheduler_type=config_dict.get('training', {}).get('lr_scheduler_type', 'cosine'),
                fp16=config_dict.get('training', {}).get('fp16', False),
                bf16=config_dict.get('training', {}).get('bf16', True),
                save_strategy=config_dict.get('training', {}).get('save_strategy', 'steps'),
                save_steps=config_dict.get('training', {}).get('save_steps', 100),
                save_total_limit=config_dict.get('training', {}).get('save_total_limit', 3),
                logging_steps=config_dict.get('training', {}).get('logging_steps', 10),
                logging_first_step=config_dict.get('training', {}).get('logging_first_step', True),
                evaluation_strategy=config_dict.get('training', {}).get('evaluation_strategy', 'steps'),
                eval_steps=config_dict.get('training', {}).get('eval_steps', 100),
                gradient_checkpointing=config_dict.get('training', {}).get('gradient_checkpointing', True),
                optim=config_dict.get('training', {}).get('optim', 'paged_adamw_32bit'),
                max_grad_norm=config_dict.get('training', {}).get('max_grad_norm', 0.3),
            ),
            dataset=DatasetConfig(
                path=config_dict.get('dataset', {}).get('path', 'mlabonne/guanaco-llama2-1k'),
                text_column=config_dict.get('dataset', {}).get('text_column', 'text'),
                instruction_column=config_dict.get('dataset', {}).get('instruction_column', 'instruction'),
                response_column=config_dict.get('dataset', {}).get('response_column', 'response'),
                max_seq_length=config_dict.get('dataset', {}).get('max_seq_length', 1024),
                train_split=config_dict.get('dataset', {}).get('train_split', 0.9),
                seed=config_dict.get('dataset', {}).get('seed', 42),
            ),
            mlflow=MLflowConfig(
                experiment_name=config_dict.get('mlflow', {}).get('experiment_name', 'llm-fine-tuning'),
                tracking_uri=config_dict.get('mlflow', {}).get('tracking_uri'),
                tags=config_dict.get('mlflow', {}).get('tags', {
                    "project": "fine-tuning-pipeline",
                    "framework": "transformers",
                    "method": "qlora"
                }),
            ),
            output=OutputConfig(
                dir=config_dict.get('output', {}).get('dir', './outputs'),
                model_name=config_dict.get('output', {}).get('model_name', 'fine-tuned-model'),
                push_to_hub=config_dict.get('output', {}).get('push_to_hub', False),
                hub_model_id=config_dict.get('output', {}).get('hub_model_id'),
            ),
        )


# Constantes de ambiente
ENV_VARS = {
    "DAGSHUB_TOKEN": os.getenv("DAGSHUB_TOKEN"),
    "DAGSHUB_USERNAME": os.getenv("DAGSHUB_USERNAME"),
    "DAGSHUB_REPO": os.getenv("DAGSHUB_REPO"),
    "HF_TOKEN": os.getenv("HF_TOKEN"),
    "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
}


def get_mlflow_tracking_uri(username: str = None, repo: str = None) -> str:
    """Gera a URI de tracking do MLflow para DagsHub."""
    username = username or ENV_VARS.get("DAGSHUB_USERNAME")
    repo = repo or ENV_VARS.get("DAGSHUB_REPO")
    
    if username and repo:
        return f"https://dagshub.com/{username}/{repo}.mlflow"
    return None
