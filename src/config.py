"""
Configurações e constantes do projeto de Fine-Tuning.
"""

import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Type, TypeVar
import yaml


# TypeVar para métodos genéricos
T = TypeVar('T')


def _get_nested(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Obtém valor aninhado de um dicionário de forma segura.
    
    Args:
        data: Dicionário fonte
        *keys: Chaves aninhadas (ex: 'model', 'quantization', 'bits')
        default: Valor padrão se não encontrar
        
    Returns:
        Valor encontrado ou default
    """
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, {})
        else:
            return default
    return data if data != {} else default


def _from_dict(cls: Type[T], data: Dict[str, Any], prefix: str = "") -> T:
    """
    Cria uma dataclass a partir de um dicionário.
    
    Args:
        cls: Classe dataclass
        data: Dicionário com dados
        prefix: Prefixo para acessar dados aninhados
        
    Returns:
        Instância da dataclass
    """
    if not data:
        return cls()
    
    # Obter os campos da dataclass
    field_names = {f.name for f in fields(cls)}
    
    # Filtrar apenas os campos válidos
    filtered_data = {k: v for k, v in data.items() if k in field_names}
    
    return cls(**filtered_data)


@dataclass
class ModelConfig:
    """Configurações do modelo."""
    name: str = "Qwen/Qwen2.5-1.5B"  # Modelo padrão (não requer autenticação)
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
        """
        Carrega configurações de um arquivo YAML.
        
        Args:
            yaml_path: Caminho para o arquivo YAML
            
        Returns:
            Instância de PipelineConfig
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        
        # Helper para acessar valores aninhados de forma segura
        def get(section: str, key: str, default: Any = None) -> Any:
            return cfg.get(section, {}).get(key, default)
        
        def get_quant(key: str, default: Any = None) -> Any:
            return cfg.get('model', {}).get('quantization', {}).get(key, default)
        
        return cls(
            model=ModelConfig(
                name=get('model', 'name', 'Qwen/Qwen2.5-1.5B'),
                quantization_enabled=get_quant('enabled', True),
                quantization_bits=get_quant('bits', 4),
                bnb_4bit_compute_dtype=get_quant('bnb_4bit_compute_dtype', 'float16'),
                bnb_4bit_quant_type=get_quant('bnb_4bit_quant_type', 'nf4'),
                bnb_4bit_use_double_quant=get_quant('bnb_4bit_use_double_quant', True),
            ),
            lora=_from_dict(LoRAConfig, cfg.get('lora', {})),
            training=_from_dict(TrainingConfig, cfg.get('training', {})),
            dataset=_from_dict(DatasetConfig, cfg.get('dataset', {})),
            mlflow=_from_dict(MLflowConfig, cfg.get('mlflow', {})),
            output=_from_dict(OutputConfig, cfg.get('output', {})),
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """
        Cria configuração a partir de um dicionário.
        
        Args:
            config_dict: Dicionário com configurações
            
        Returns:
            Instância de PipelineConfig
        """
        return cls(
            model=_from_dict(ModelConfig, config_dict.get('model', {})),
            lora=_from_dict(LoRAConfig, config_dict.get('lora', {})),
            training=_from_dict(TrainingConfig, config_dict.get('training', {})),
            dataset=_from_dict(DatasetConfig, config_dict.get('dataset', {})),
            mlflow=_from_dict(MLflowConfig, config_dict.get('mlflow', {})),
            output=_from_dict(OutputConfig, config_dict.get('output', {})),
        )


# Constantes de ambiente
ENV_VARS = {
    "DAGSHUB_TOKEN": os.getenv("DAGSHUB_TOKEN"),
    "DAGSHUB_USERNAME": os.getenv("DAGSHUB_USERNAME"),
    "DAGSHUB_REPO": os.getenv("DAGSHUB_REPO"),
    "HF_TOKEN": os.getenv("HF_TOKEN"),
    "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
}


def get_mlflow_tracking_uri(username: Optional[str] = None, repo: Optional[str] = None) -> Optional[str]:
    """Gera a URI de tracking do MLflow para DagsHub."""
    username = username or ENV_VARS.get("DAGSHUB_USERNAME")
    repo = repo or ENV_VARS.get("DAGSHUB_REPO")
    
    if username and repo:
        return f"https://dagshub.com/{username}/{repo}.mlflow"
    return None
