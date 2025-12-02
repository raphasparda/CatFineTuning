"""
Trainer para Fine-Tuning de LLMs com QLoRA.
"""

import glob
import os
import logging
from typing import Dict, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
from trl import SFTTrainer
from datasets import Dataset

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from .config import PipelineConfig

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class FineTuningTrainer:
    """Trainer para fine-tuning de LLMs com QLoRA."""
    
    def __init__(self, config: PipelineConfig):
        """
        Inicializa o trainer.
        
        Args:
            config: Configuração do pipeline
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._is_trained = False
        
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Retorna configuração de quantização se habilitada."""
        if not self.config.model.quantization_enabled:
            return None
            
        # Detectar dtype suportado
        compute_dtype = self._get_compute_dtype()
        
        return BitsAndBytesConfig(
            load_in_4bit=(self.config.model.quantization_bits == 4),
            load_in_8bit=(self.config.model.quantization_bits == 8),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=self.config.model.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.model.bnb_4bit_use_double_quant,
        )
    
    def _get_compute_dtype(self) -> torch.dtype:
        """Detecta o dtype de computação suportado pela GPU."""
        dtype_str = self.config.model.bnb_4bit_compute_dtype
        
        if dtype_str == "float16":
            return torch.float16
        elif dtype_str == "bfloat16":
            # Verificar se bf16 é suportado
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            else:
                logger.warning("bfloat16 não suportado, usando float16")
                return torch.float16
        else:
            return torch.float32
    
    def _get_lora_config(self) -> LoraConfig:
        """Retorna configuração do LoRA."""
        return LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.config.lora.target_modules,
        )
    
    def _get_training_arguments(self) -> TrainingArguments:
        """Retorna argumentos de treinamento."""
        # Detectar suporte a bf16/fp16
        use_bf16 = False
        use_fp16 = False
        
        if torch.cuda.is_available():
            if self.config.training.bf16 and torch.cuda.is_bf16_supported():
                use_bf16 = True
            elif self.config.training.fp16:
                use_fp16 = True
            else:
                # Fallback para fp16 se nenhum especificado
                use_fp16 = True
        
        return TrainingArguments(
            output_dir=self.config.output.dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_ratio=self.config.training.warmup_ratio,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            fp16=use_fp16,
            bf16=use_bf16,
            save_strategy=self.config.training.save_strategy,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            logging_steps=self.config.training.logging_steps,
            logging_first_step=self.config.training.logging_first_step,
            evaluation_strategy=self.config.training.evaluation_strategy,
            eval_steps=self.config.training.eval_steps,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            optim=self.config.training.optim,
            max_grad_norm=self.config.training.max_grad_norm,
            report_to="mlflow" if MLFLOW_AVAILABLE and self.config.mlflow.tracking_uri else "none",
            load_best_model_at_end=True,  # Carregar melhor modelo ao final
            metric_for_best_model="eval_loss",  # Métrica para early stopping
            greater_is_better=False,  # Menor loss é melhor
        )
    
    def load_model(self, model_name: Optional[str] = None) -> None:
        """
        Carrega o modelo base com quantização.
        
        Args:
            model_name: Nome do modelo (opcional, usa config se não fornecido)
        """
        model_name = model_name or self.config.model.name
        logger.info(f"Carregando modelo: {model_name}")
        
        # Carregar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        # Garantir pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Configuração de quantização
        bnb_config = self._get_quantization_config()
        
        # Carregar modelo
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Preparar para k-bit training
        if bnb_config is not None:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Aplicar LoRA
        lora_config = self._get_lora_config()
        self.model = get_peft_model(self.model, lora_config)
        
        # Log info
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        memory_gb = self.model.get_memory_footprint() / (1024**3)
        
        logger.info(f"Modelo carregado! Memória: {memory_gb:.2f} GB")
        logger.info(f"Parâmetros treináveis: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    def setup_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        text_field: str = "text",
        early_stopping_patience: Optional[int] = 3,
    ) -> None:
        """
        Configura o SFTTrainer.
        
        Args:
            train_dataset: Dataset de treino
            eval_dataset: Dataset de avaliação (opcional)
            text_field: Nome do campo de texto no dataset
            early_stopping_patience: Número de avaliações sem melhoria antes de parar
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Modelo não carregado. Execute load_model() primeiro.")
        
        training_args = self._get_training_arguments()
        
        # Configurar callbacks
        callbacks = []
        
        # Early stopping (apenas se tiver dataset de avaliação)
        if eval_dataset is not None and early_stopping_patience:
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
            )
            logger.info(f"Early stopping habilitado (patience={early_stopping_patience})")
        
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            tokenizer=self.tokenizer,
            dataset_text_field=text_field,
            max_seq_length=self.config.dataset.max_seq_length,
            packing=False,
            callbacks=callbacks if callbacks else None,
        )
        
        logger.info("Trainer configurado!")
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict:
        """
        Executa o treinamento.
        
        Args:
            resume_from_checkpoint: Caminho para checkpoint para retomar treinamento
        
        Returns:
            Métricas de treinamento
        """
        if self.trainer is None:
            raise RuntimeError("Trainer não configurado. Execute setup_trainer() primeiro.")
        
        # Detectar checkpoint automaticamente se não especificado
        if resume_from_checkpoint is None:
            resume_from_checkpoint = self._find_latest_checkpoint()
        
        if resume_from_checkpoint:
            logger.info(f"Retomando treinamento de: {resume_from_checkpoint}")
        else:
            logger.info("Iniciando treinamento...")
        
        # MLflow tracking
        if MLFLOW_AVAILABLE and self.config.mlflow.tracking_uri:
            with mlflow.start_run():
                # Log parâmetros
                mlflow.log_params({
                    "model_name": self.config.model.name,
                    "lora_r": self.config.lora.r,
                    "lora_alpha": self.config.lora.lora_alpha,
                    "learning_rate": self.config.training.learning_rate,
                    "epochs": self.config.training.num_train_epochs,
                    "batch_size": self.config.training.per_device_train_batch_size,
                })
                
                result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
                
                # Log métricas finais
                if result.metrics:
                    mlflow.log_metrics(result.metrics)
        else:
            result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        self._is_trained = True
        logger.info("Treinamento concluído!")
        
        return result.metrics if result.metrics else {}
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """
        Encontra o checkpoint mais recente no diretório de output.
        
        Returns:
            Caminho do checkpoint ou None
        """
        output_dir = self.config.output.dir
        checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
        checkpoints = glob.glob(checkpoint_pattern)
        
        if not checkpoints:
            return None
        
        # Ordenar por número do checkpoint
        def get_checkpoint_number(path: str) -> int:
            try:
                return int(path.split("-")[-1])
            except ValueError:
                return 0
        
        checkpoints.sort(key=get_checkpoint_number)
        latest = checkpoints[-1]
        
        logger.info(f"Checkpoint encontrado: {latest}")
        return latest
    
    def save_model(self, output_path: Optional[str] = None) -> str:
        """
        Salva o modelo treinado.
        
        Args:
            output_path: Caminho de saída (opcional)
            
        Returns:
            Caminho onde o modelo foi salvo
        """
        if not self._is_trained:
            logger.warning("Modelo não foi treinado. Salvando estado atual...")
        
        output_path = output_path or os.path.join(
            self.config.output.dir,
            self.config.output.model_name
        )
        
        # Criar diretório se não existir
        os.makedirs(output_path, exist_ok=True)
        
        # Salvar modelo e tokenizer
        self.trainer.save_model(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        logger.info(f"Modelo salvo em: {output_path}")
        return output_path
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Gera texto a partir de um prompt.
        
        Args:
            prompt: Texto de entrada
            max_new_tokens: Máximo de tokens a gerar
            temperature: Temperatura de amostragem
            top_p: Top-p para nucleus sampling
            do_sample: Se deve usar amostragem
            
        Returns:
            Texto gerado
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Modelo não carregado.")
        
        # Preparar input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Gerar
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decodificar apenas tokens novos
        generated = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )
        
        return generated
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[PipelineConfig] = None,
    ) -> "FineTuningTrainer":
        """
        Carrega um modelo previamente treinado.
        
        Args:
            model_path: Caminho do modelo salvo
            config: Configuração (opcional)
            
        Returns:
            Instância do trainer com modelo carregado
        """
        config = config or PipelineConfig()
        trainer = cls(config)
        
        # Carregar tokenizer
        trainer.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Carregar modelo base + adapters
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            device_map="auto",
            trust_remote_code=True,
        )
        
        trainer.model = PeftModel.from_pretrained(base_model, model_path)
        trainer._is_trained = True
        
        logger.info(f"Modelo carregado de: {model_path}")
        return trainer
    
    def merge_and_save(
        self,
        output_dir: str,
        safe_serialization: bool = True,
    ) -> str:
        """
        Faz merge dos adapters LoRA no modelo base e salva.
        
        Útil para inferência mais rápida em produção, pois não
        precisa carregar os adapters separadamente.
        
        Args:
            output_dir: Diretório para salvar o modelo merged
            safe_serialization: Usar safetensors (mais seguro)
            
        Returns:
            Caminho do modelo merged
        """
        if not self._is_trained:
            raise ValueError("Modelo ainda não foi treinado!")
        
        if self.model is None:
            raise ValueError("Modelo não está carregado!")
        
        logger.info("Fazendo merge dos adapters LoRA no modelo base...")
        
        # Merge adapters no modelo base
        merged_model = self.model.merge_and_unload()
        
        # Criar diretório se não existir
        os.makedirs(output_dir, exist_ok=True)
        
        # Salvar modelo merged
        merged_model.save_pretrained(
            output_dir,
            safe_serialization=safe_serialization,
        )
        
        # Salvar tokenizer junto
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Modelo merged salvo em: {output_dir}")
        return output_dir
