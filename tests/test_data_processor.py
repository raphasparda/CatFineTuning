"""
Testes unitários para o módulo de processamento de dados.
"""

import json
import os
import tempfile
import pytest

from src.data_processor import DataProcessor, create_sample_dataset


class MockTokenizer:
    """Mock do tokenizer para testes."""
    
    def __init__(self):
        self.pad_token = "[PAD]"
        self.eos_token = "[EOS]"
    
    def __call__(
        self, 
        texts, 
        truncation=True, 
        max_length=512, 
        padding="max_length",
        return_tensors=None,
    ):
        """Simula tokenização."""
        results = {
            "input_ids": [],
            "attention_mask": [],
        }
        
        for text in texts:
            # Simular tokenização simples
            tokens = list(range(min(len(text.split()), max_length)))
            padding_len = max_length - len(tokens)
            
            results["input_ids"].append(tokens + [0] * padding_len)
            results["attention_mask"].append([1] * len(tokens) + [0] * padding_len)
        
        return results


class TestDataProcessor:
    """Testes para DataProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Cria um processador para testes."""
        tokenizer = MockTokenizer()
        return DataProcessor(tokenizer, max_seq_length=128)
    
    @pytest.fixture
    def sample_data(self):
        """Dados de exemplo."""
        return [
            {
                "instruction": "Explique Python",
                "response": "Python é uma linguagem de programação",
                "input": "",
            },
            {
                "instruction": "O que é ML?",
                "response": "Machine Learning é...",
                "input": "Contexto adicional",
            },
        ]
    
    def test_format_prompt_without_input(self, processor):
        """Testa formatação de prompt sem input."""
        example = {
            "instruction": "Teste",
            "response": "Resposta",
        }
        
        prompt = processor.format_prompt(example)
        
        assert "Teste" in prompt
        assert "Resposta" in prompt
        assert "<|begin_of_text|>" in prompt
    
    def test_format_prompt_with_input(self, processor):
        """Testa formatação de prompt com input."""
        example = {
            "instruction": "Teste",
            "response": "Resposta",
            "input": "Input adicional",
        }
        
        prompt = processor.format_prompt(example)
        
        assert "Teste" in prompt
        assert "Input adicional" in prompt
        assert "Resposta" in prompt
    
    def test_tokenizer_pad_token_fallback(self):
        """Testa fallback de pad_token para eos_token."""
        class TokenizerNoPad:
            pad_token = None
            eos_token = "[EOS]"
        
        tokenizer = TokenizerNoPad()
        processor = DataProcessor(tokenizer)
        
        assert processor.tokenizer.pad_token == "[EOS]"
    
    def test_load_jsonl_valid(self, processor, sample_data):
        """Testa carregamento de JSONL válido."""
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.jsonl', 
            delete=False,
            encoding='utf-8'
        ) as f:
            for item in sample_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            temp_path = f.name
        
        try:
            dataset = processor.load_jsonl(temp_path)
            
            assert len(dataset) == 2
            assert "instruction" in dataset.column_names
            assert "response" in dataset.column_names
        finally:
            os.unlink(temp_path)
    
    def test_load_jsonl_file_not_found(self, processor):
        """Testa erro quando arquivo não existe."""
        with pytest.raises(FileNotFoundError):
            processor.load_jsonl("/caminho/inexistente.jsonl")
    
    def test_load_jsonl_empty_file(self, processor):
        """Testa erro com arquivo vazio."""
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.jsonl', 
            delete=False
        ) as f:
            f.write("")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="vazio"):
                processor.load_jsonl(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_jsonl_invalid_json(self, processor):
        """Testa erro com JSON inválido."""
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.jsonl', 
            delete=False
        ) as f:
            f.write('{"valid": "json"}\n')
            f.write('invalid json line\n')
            f.write('{"another": "valid"}\n')
            temp_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                processor.load_jsonl(temp_path)
        finally:
            os.unlink(temp_path)


class TestCreateSampleDataset:
    """Testes para create_sample_dataset."""
    
    def test_creates_correct_number_of_samples(self):
        """Testa se cria o número correto de amostras."""
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.jsonl', 
            delete=False
        ) as f:
            temp_path = f.name
        
        try:
            create_sample_dataset(temp_path, num_samples=50)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                lines = [line for line in f if line.strip()]
            
            assert len(lines) == 50
        finally:
            os.unlink(temp_path)
    
    def test_creates_valid_jsonl(self):
        """Testa se o JSONL criado é válido."""
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.jsonl', 
            delete=False
        ) as f:
            temp_path = f.name
        
        try:
            create_sample_dataset(temp_path, num_samples=10)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        assert "instruction" in data
                        assert "response" in data
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
