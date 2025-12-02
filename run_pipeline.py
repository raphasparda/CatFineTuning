#!/usr/bin/env python3
"""
Script principal para executar o pipeline de fine-tuning.

Uso:
    python run_pipeline.py --config configs/training_config.yaml
    python run_pipeline.py --model "Qwen/Qwen2.5-1.5B" --dataset "data/sample_dataset.jsonl"
"""

import argparse
import os
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import PipelineConfig
from src.data_processor import DataProcessor
from src.trainer import FineTuningTrainer
from src.utils import (
    setup_environment,
    check_gpu,
    print_gpu_info,
    print_banner,
    Timer,
    get_execution_environment,
    validate_credentials,
)


def parse_args():
    """Parse argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="PurrTune - Pipeline de Fine-Tuning de LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python run_pipeline.py --config configs/training_config.yaml
  python run_pipeline.py --model "Qwen/Qwen2.5-1.5B" --epochs 3
  python run_pipeline.py --dataset data/devops_dataset.jsonl --output ./my_model
        """
    )
    
    # Configuração
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Caminho para arquivo de configuração YAML"
    )
    
    # Modelo
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Nome do modelo (ex: Qwen/Qwen2.5-1.5B)"
    )
    
    # Dataset
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="Caminho do dataset (JSONL) ou nome do HuggingFace Hub"
    )
    
    # Treinamento
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Número de épocas de treinamento"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Tamanho do batch por dispositivo"
    )
    
    parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        default=None,
        help="Taxa de aprendizado"
    )
    
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Tamanho máximo da sequência"
    )
    
    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Diretório de saída para o modelo"
    )
    
    # Flags
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Desabilitar quantização 4-bit"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reprodutibilidade"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas mostrar configuração sem executar"
    )
    
    return parser.parse_args()


def load_config(args) -> PipelineConfig:
    """Carrega configuração combinando arquivo YAML e argumentos CLI."""
    
    # Carregar config base
    if args.config and os.path.exists(args.config):
        config = PipelineConfig.from_yaml(args.config)
        print(f"[OK] Config carregada: {args.config}")
    else:
        config = PipelineConfig()
        print("[OK] Usando configuração padrão")
    
    # Override com argumentos CLI
    if args.model:
        config.model.name = args.model
    
    if args.dataset:
        config.dataset.path = args.dataset
    
    if args.epochs:
        config.training.num_train_epochs = args.epochs
    
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    if args.max_seq_length:
        config.dataset.max_seq_length = args.max_seq_length
    
    if args.output:
        config.output.dir = args.output
    
    if args.no_quantization:
        config.model.quantization_enabled = False
    
    config.dataset.seed = args.seed
    
    return config


def print_config(config: PipelineConfig) -> None:
    """Imprime configuração de forma formatada."""
    print("\n" + "="*60)
    print("CONFIGURAÇÃO DO PIPELINE")
    print("="*60)
    
    print("\n[MODELO]")
    print(f"  Nome: {config.model.name}")
    print(f"  Quantização: {'4-bit' if config.model.quantization_enabled else 'Desabilitada'}")
    
    print("\n[LORA]")
    print(f"  r: {config.lora.r}")
    print(f"  alpha: {config.lora.lora_alpha}")
    print(f"  dropout: {config.lora.lora_dropout}")
    
    print("\n[TREINAMENTO]")
    print(f"  Épocas: {config.training.num_train_epochs}")
    print(f"  Batch size: {config.training.per_device_train_batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
    
    print("\n[DATASET]")
    print(f"  Path: {config.dataset.path}")
    print(f"  Max seq length: {config.dataset.max_seq_length}")
    print(f"  Train split: {config.dataset.train_split}")
    
    print("\n[OUTPUT]")
    print(f"  Diretório: {config.output.dir}")
    print("="*60 + "\n")


def validate_credentials_check(config: PipelineConfig) -> bool:
    """
    Valida credenciais necessárias para o pipeline.
    
    Args:
        config: Configuração do pipeline
        
    Returns:
        True se credenciais suficientes, False caso contrário
    """
    creds = validate_credentials(require_hf=False, require_dagshub=False)
    
    print("\n[CREDENCIAIS]")
    print(f"  HuggingFace Token: {'[OK] Configurado' if creds['hf_token'] else '[--] Não configurado'}")
    print(f"  DagsHub Token: {'[OK] Configurado' if creds['dagshub_token'] else '[--] Não configurado'}")
    print(f"  DagsHub Username: {'[OK] Configurado' if creds.get('dagshub_username') else '[--] Não configurado'}")
    print(f"  DagsHub Repo: {'[OK] Configurado' if creds.get('dagshub_repo') else '[--] Não configurado'}")
    
    # Verificar se modelo requer autenticação HF
    model_requires_auth = any(
        prefix in config.model.name.lower() 
        for prefix in ['meta-llama', 'llama', 'mistral-community']
    )
    
    if model_requires_auth and not creds['hf_token']:
        print(f"\n[WARN] O modelo '{config.model.name}' pode requerer autenticação HF!")
        print("       Configure HF_TOKEN ou use um modelo aberto (ex: Qwen/Qwen2.5-1.5B)")
        return False
    
    # Verificar DagsHub para MLflow tracking
    if config.mlflow.tracking_uri and 'dagshub' in str(config.mlflow.tracking_uri):
        if not creds['dagshub_token']:
            print("\n[WARN] MLflow tracking com DagsHub configurado mas sem credenciais!")
            print("       Configure DAGSHUB_TOKEN, DAGSHUB_USERNAME e DAGSHUB_REPO")
    
    return True


def main():
    """Função principal do pipeline."""
    
    # Banner
    print_banner("[=^.^=] PurrTune - Fine-Tuning Pipeline")
    
    # Parse argumentos
    args = parse_args()
    
    # Detectar ambiente
    env = get_execution_environment()
    print(f"\n[ENV] Ambiente: {env}")
    
    # Verificar GPU
    if not check_gpu():
        print("[WARN] Sem GPU! O treinamento será muito lento.")
        response = input("Continuar mesmo assim? (s/N): ")
        if response.lower() != 's':
            print("Abortando.")
            return 1
    else:
        print_gpu_info()
    
    # Carregar configuração
    config = load_config(args)
    print_config(config)
    
    # Validar credenciais
    if not validate_credentials_check(config):
        print("\n[ERRO] Credenciais insuficientes para o modelo selecionado.")
        print("       Use --model para escolher outro modelo ou configure as credenciais.")
        return 1
    
    # Dry run - apenas mostrar config
    if args.dry_run:
        print("\n[DRY-RUN] Configuração validada. Nenhum treinamento executado.")
        return 0
    
    # Setup ambiente
    setup_environment(seed=args.seed)
    
    # ========== PIPELINE ==========
    
    with Timer("Pipeline completo"):
        
        # 1. Carregar modelo
        print("\n[1/4] Carregando modelo...")
        trainer = FineTuningTrainer(config)
        
        with Timer("Carregamento do modelo"):
            trainer.load_model()
        
        # 2. Preparar dataset
        print("\n[2/4] Preparando dataset...")
        data_processor = DataProcessor(
            tokenizer=trainer.tokenizer,
            max_seq_length=config.dataset.max_seq_length,
            instruction_column=config.dataset.instruction_column,
            response_column=config.dataset.response_column,
        )
        
        with Timer("Processamento do dataset"):
            # Verificar se é JSONL local ou HuggingFace
            if config.dataset.path.endswith('.jsonl'):
                raw_dataset = data_processor.load_jsonl(config.dataset.path)
            else:
                raw_dataset = data_processor.load_dataset(config.dataset.path, split="train")
            
            datasets = data_processor.prepare_dataset(
                raw_dataset,
                train_split=config.dataset.train_split,
                seed=config.dataset.seed,
            )
        
        print(f"  Train: {len(datasets['train'])} amostras")
        print(f"  Eval: {len(datasets['eval'])} amostras")
        
        # 3. Treinar
        print("\n[3/4] Treinando...")
        trainer.setup_trainer(
            train_dataset=datasets['train'],
            eval_dataset=datasets['eval'],
        )
        
        with Timer("Treinamento"):
            metrics = trainer.train()
        
        print(f"  Loss final: {metrics.get('train_loss', 'N/A')}")
        
        # 4. Salvar
        print("\n[4/4] Salvando modelo...")
        output_path = trainer.save_model()
        
    # ========== FIM ==========
    
    print("\n" + "="*60)
    print("[OK] PIPELINE CONCLUÍDO COM SUCESSO!")
    print(f"[OK] Modelo salvo em: {output_path}")
    print("="*60)
    
    # Teste rápido
    print("\n[TESTE] Gerando resposta de exemplo...")
    try:
        test_prompt = "O que é Machine Learning?"
        response = trainer.generate(test_prompt, max_new_tokens=100)
        print(f"  Q: {test_prompt}")
        print(f"  A: {response[:200]}...")
    except Exception as e:
        print(f"  [WARN] Erro no teste: {e}")
    
    return 0


if __name__ == "__main__":
    exit(main())
