"""
Processamento de dados para Fine-Tuning de LLMs.
"""

import json
from typing import Dict, List, Optional, Union
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


class DataProcessor:
    """Processador de dados para fine-tuning."""
    
    # Template de prompt para LLaMA 3 (formato de chat)
    PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response}<|eot_id|>"""

    PROMPT_TEMPLATE_WITH_INPUT = """<|begin_of_text|><|start_header_id|>user<|end_header_id|}

{instruction}

{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response}<|eot_id|>"""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 512,
        instruction_column: str = "instruction",
        response_column: str = "response",
        input_column: str = "input",
    ):
        """
        Inicializa o processador de dados.
        
        Args:
            tokenizer: Tokenizer do modelo
            max_seq_length: Tamanho maximo da sequencia
            instruction_column: Nome da coluna de instrucao
            response_column: Nome da coluna de resposta
            input_column: Nome da coluna de input adicional
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.instruction_column = instruction_column
        self.response_column = response_column
        self.input_column = input_column
        
        # Garantir que o tokenizer tem pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def format_prompt(self, example: Dict) -> str:
        """
        Formata um exemplo no template de prompt.
        
        Args:
            example: Dicionario com instrucao e resposta
            
        Returns:
            Texto formatado
        """
        instruction = example.get(self.instruction_column, "")
        response = example.get(self.response_column, "")
        input_text = example.get(self.input_column, "")
        
        if input_text:
            return self.PROMPT_TEMPLATE_WITH_INPUT.format(
                instruction=instruction,
                input=input_text,
                response=response
            )
        
        return self.PROMPT_TEMPLATE.format(
            instruction=instruction,
            response=response
        )
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """
        Tokeniza os exemplos.
        
        Args:
            examples: Batch de exemplos
            
        Returns:
            Exemplos tokenizados
        """
        # Formatar prompts
        if isinstance(examples.get(self.instruction_column), list):
            texts = []
            for i in range(len(examples[self.instruction_column])):
                example = {
                    self.instruction_column: examples[self.instruction_column][i],
                    self.response_column: examples[self.response_column][i],
                    self.input_column: examples.get(self.input_column, [""] * len(examples[self.instruction_column]))[i]
                }
                texts.append(self.format_prompt(example))
        else:
            texts = [self.format_prompt(examples)]
        
        # Tokenizar
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        
        # Labels sao os mesmos que input_ids para language modeling
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def load_jsonl(self, file_path: str) -> Dataset:
        """
        Carrega dataset de arquivo JSONL.
        
        Args:
            file_path: Caminho do arquivo
            
        Returns:
            Dataset do Hugging Face
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        return Dataset.from_list(data)
    
    def load_dataset(
        self,
        path: str,
        split: Optional[str] = None,
        **kwargs
    ) -> Dataset:
        """
        Carrega dataset de varias fontes.
        
        Args:
            path: Caminho do arquivo ou nome do dataset HF
            split: Split do dataset (train, test, etc.)
            **kwargs: Argumentos adicionais
            
        Returns:
            Dataset carregado
        """
        if path.endswith('.jsonl') or path.endswith('.json'):
            return self.load_jsonl(path)
        else:
            # Carregar do Hugging Face Hub
            return load_dataset(path, split=split, **kwargs)
    
    def prepare_dataset(
        self,
        dataset: Union[Dataset, str],
        train_split: float = 0.9,
        seed: int = 42,
    ) -> Dict[str, Dataset]:
        """
        Prepara o dataset para treinamento.
        
        Args:
            dataset: Dataset ou caminho
            train_split: Proporcao para treino
            seed: Seed para reprodutibilidade
            
        Returns:
            Dicionario com datasets de treino e validacao
        """
        # Carregar se for caminho
        if isinstance(dataset, str):
            dataset = self.load_dataset(dataset)
        
        # Tokenizar
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizando dataset"
        )
        
        # Split train/eval
        split_dataset = tokenized_dataset.train_test_split(
            train_size=train_split,
            seed=seed
        )
        
        return {
            "train": split_dataset["train"],
            "eval": split_dataset["test"]
        }


def create_sample_dataset(output_path: str, num_samples: int = 100):
    """
    Cria um dataset de exemplo para testes.
    
    Args:
        output_path: Caminho de saida
        num_samples: Numero de amostras
    """
    samples = [
        {
            "instruction": "Explique o que e Machine Learning em termos simples.",
            "response": "Machine Learning e uma area da inteligencia artificial que permite que computadores aprendam padroes a partir de dados, sem serem explicitamente programados. E como ensinar uma crianca: voce mostra exemplos e ela aprende a reconhecer padroes sozinha."
        },
        {
            "instruction": "Qual e a diferenca entre Python e JavaScript?",
            "response": "Python e uma linguagem focada em legibilidade e versatilidade, muito usada em ciencia de dados e backend. JavaScript foi criada para navegadores web e e essencial para desenvolvimento frontend, mas tambem funciona no backend com Node.js."
        },
        {
            "instruction": "Como funciona uma rede neural?",
            "response": "Uma rede neural e inspirada no cerebro humano. Ela possui camadas de 'neuronios' artificiais que processam informacoes. Cada neuronio recebe inputs, aplica pesos e uma funcao de ativacao, e passa o resultado adiante. O treinamento ajusta esses pesos para melhorar as previsoes."
        },
        {
            "instruction": "O que e Fine-Tuning de modelos?",
            "response": "Fine-Tuning e o processo de pegar um modelo pre-treinado e ajusta-lo para uma tarefa especifica. Em vez de treinar do zero, voce aproveita o conhecimento ja adquirido e apenas adapta para seu caso de uso, economizando tempo e recursos computacionais."
        },
        {
            "instruction": "Explique o conceito de API REST.",
            "response": "REST (Representational State Transfer) e um estilo arquitetural para APIs web. Usa metodos HTTP (GET, POST, PUT, DELETE) para operacoes CRUD, e stateless (cada requisicao e independente), e trabalha com recursos identificados por URLs. E o padrao mais comum para APIs web."
        },
        {
            "instruction": "O que e Docker e para que serve?",
            "response": "Docker e uma plataforma de containerizacao que empacota aplicacoes com todas suas dependencias em containers isolados. Isso garante que o software funcione da mesma forma em qualquer ambiente, facilitando deploy e eliminando o famoso 'funciona na minha maquina'."
        },
        {
            "instruction": "Como funciona o Git?",
            "response": "Git e um sistema de controle de versao distribuido. Ele rastreia mudancas em arquivos, permite criar branches para trabalhar em paralelo, fazer merge para juntar alteracoes, e manter historico completo. Cada desenvolvedor tem uma copia completa do repositorio."
        },
        {
            "instruction": "O que e LoRA em LLMs?",
            "response": "LoRA (Low-Rank Adaptation) e uma tecnica eficiente de fine-tuning que congela os pesos do modelo original e adiciona pequenas matrizes treinaveis. Isso reduz drasticamente o numero de parametros a treinar (as vezes 99%+), economizando memoria e tempo."
        },
        {
            "instruction": "Explique Quantizacao de modelos.",
            "response": "Quantizacao reduz a precisao dos pesos do modelo (ex: de 32-bit para 4-bit). Isso diminui o uso de memoria e acelera inferencia, com perda minima de qualidade. E essencial para rodar modelos grandes em hardware limitado como GPUs consumer."
        },
        {
            "instruction": "O que e MLOps?",
            "response": "MLOps (Machine Learning Operations) combina ML com praticas DevOps. Inclui versionamento de dados e modelos, pipelines automatizados de treinamento, monitoramento em producao, e CI/CD para ML. O objetivo e tornar ML reproduzivel, escalavel e confiavel."
        },
    ]
    
    # Replicar para ter mais amostras
    all_samples = []
    for i in range(num_samples // len(samples) + 1):
        for sample in samples:
            all_samples.append(sample)
            if len(all_samples) >= num_samples:
                break
        if len(all_samples) >= num_samples:
            break
    
    # Salvar como JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples[:num_samples]:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"[OK] Dataset de exemplo criado com {num_samples} amostras em: {output_path}")


if __name__ == "__main__":
    # Criar dataset de exemplo
    create_sample_dataset("data/sample_dataset.jsonl", num_samples=100)
