"""
Testes unitários para o módulo de configuração.
"""

import os
import tempfile
import pytest
import yaml

from src.config import (
    ModelConfig,
    LoRAConfig,
    TrainingConfig,
    DatasetConfig,
    MLflowConfig,
    OutputConfig,
    PipelineConfig,
    get_mlflow_tracking_uri,
)


class TestModelConfig:
    """Testes para ModelConfig."""
    
    def test_default_values(self):
        """Testa valores padrão."""
        config = ModelConfig()
        
        assert config.name == "Qwen/Qwen2.5-1.5B"
        assert config.quantization_enabled is True
        assert config.quantization_bits == 4
        assert config.bnb_4bit_compute_dtype == "float16"
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_use_double_quant is True
    
    def test_custom_values(self):
        """Testa valores customizados."""
        config = ModelConfig(
            name="custom/model",
            quantization_enabled=False,
            quantization_bits=8,
        )
        
        assert config.name == "custom/model"
        assert config.quantization_enabled is False
        assert config.quantization_bits == 8


class TestLoRAConfig:
    """Testes para LoRAConfig."""
    
    def test_default_values(self):
        """Testa valores padrão."""
        config = LoRAConfig()
        
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert config.bias == "none"
        assert config.task_type == "CAUSAL_LM"
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules
    
    def test_custom_target_modules(self):
        """Testa módulos alvo customizados."""
        config = LoRAConfig(target_modules=["query", "key"])
        
        assert config.target_modules == ["query", "key"]


class TestTrainingConfig:
    """Testes para TrainingConfig."""
    
    def test_default_values(self):
        """Testa valores padrão."""
        config = TrainingConfig()
        
        assert config.num_train_epochs == 3
        assert config.per_device_train_batch_size == 4
        assert config.gradient_accumulation_steps == 4
        assert config.learning_rate == 2e-4
        assert config.gradient_checkpointing is True
    
    def test_effective_batch_size(self):
        """Testa cálculo de batch size efetivo."""
        config = TrainingConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
        )
        
        effective_batch_size = (
            config.per_device_train_batch_size * 
            config.gradient_accumulation_steps
        )
        assert effective_batch_size == 16


class TestDatasetConfig:
    """Testes para DatasetConfig."""
    
    def test_default_values(self):
        """Testa valores padrão."""
        config = DatasetConfig()
        
        assert config.max_seq_length == 1024
        assert config.train_split == 0.9
        assert config.seed == 42


class TestPipelineConfig:
    """Testes para PipelineConfig."""
    
    def test_default_config(self):
        """Testa configuração padrão completa."""
        config = PipelineConfig()
        
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.lora, LoRAConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.dataset, DatasetConfig)
        assert isinstance(config.mlflow, MLflowConfig)
        assert isinstance(config.output, OutputConfig)
    
    def test_from_yaml(self):
        """Testa carregamento de YAML."""
        yaml_content = {
            "model": {
                "name": "test/model",
                "quantization": {
                    "enabled": True,
                    "bits": 4,
                }
            },
            "lora": {
                "r": 32,
                "lora_alpha": 64,
            },
            "training": {
                "num_train_epochs": 5,
            }
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.yaml', 
            delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name
        
        try:
            config = PipelineConfig.from_yaml(temp_path)
            
            assert config.model.name == "test/model"
            assert config.lora.r == 32
            assert config.lora.lora_alpha == 64
            assert config.training.num_train_epochs == 5
        finally:
            os.unlink(temp_path)


class TestGetMlflowTrackingUri:
    """Testes para get_mlflow_tracking_uri."""
    
    def test_with_valid_params(self):
        """Testa com parâmetros válidos."""
        uri = get_mlflow_tracking_uri("user", "repo")
        
        assert uri == "https://dagshub.com/user/repo.mlflow"
    
    def test_with_missing_params(self):
        """Testa com parâmetros faltando."""
        uri = get_mlflow_tracking_uri(None, None)
        
        assert uri is None
    
    def test_with_partial_params(self):
        """Testa com parâmetros parciais."""
        uri = get_mlflow_tracking_uri("user", None)
        
        assert uri is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
