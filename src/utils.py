"""
Utilitários para o Pipeline de Fine-Tuning.
"""

import os
import time
import logging
from typing import Optional, Dict, Any

import torch

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def is_colab() -> bool:
    """Verifica se está rodando no Google Colab."""
    try:
        import importlib.util
        return importlib.util.find_spec("google.colab") is not None
    except ImportError:
        return False


def is_kaggle() -> bool:
    """Verifica se está rodando no Kaggle."""
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None


def get_execution_environment() -> str:
    """
    Detecta o ambiente de execução.
    
    Returns:
        Nome do ambiente: 'colab', 'kaggle', 'local'
    """
    if is_colab():
        return 'colab'
    elif is_kaggle():
        return 'kaggle'
    else:
        return 'local'


def check_gpu() -> bool:
    """
    Verifica se GPU está disponível.
    
    Returns:
        True se GPU disponível, False caso contrário
    """
    if torch.cuda.is_available():
        return True
    else:
        logger.warning("GPU não disponível! O treinamento será muito lento.")
        return False


def print_gpu_info() -> Dict[str, Any]:
    """
    Imprime informações detalhadas da GPU.
    
    Returns:
        Dicionário com informações da GPU
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": []
    }
    
    if not torch.cuda.is_available():
        logger.warning("CUDA não disponível")
        return info
    
    info["device_count"] = torch.cuda.device_count()
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        device_info = {
            "index": i,
            "name": props.name,
            "total_memory_gb": props.total_memory / (1024**3),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        }
        info["devices"].append(device_info)
        
        logger.info(f"GPU {i}: {props.name}")
        logger.info(f"  Memória: {device_info['total_memory_gb']:.1f} GB")
        logger.info(f"  Compute Capability: {device_info['compute_capability']}")
    
    # Verificar suporte a bf16
    if torch.cuda.is_bf16_supported():
        logger.info("  bfloat16: Suportado")
        info["bf16_supported"] = True
    else:
        logger.info("  bfloat16: Não suportado (usando float16)")
        info["bf16_supported"] = False
    
    return info


def setup_environment(
    seed: int = 42,
    deterministic: bool = True,
    hf_token: Optional[str] = None,
) -> None:
    """
    Configura o ambiente de execução.
    
    Args:
        seed: Seed para reprodutibilidade
        deterministic: Se deve usar operações determinísticas
        hf_token: Token do Hugging Face (opcional)
    """
    import random
    import numpy as np
    
    # Seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # Variáveis de ambiente
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Hugging Face token
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            logger.info("Autenticado no Hugging Face")
        except Exception as e:
            logger.warning(f"Erro ao autenticar no HF: {e}")
    
    logger.info(f"Ambiente configurado (seed={seed})")


def setup_mlflow_dagshub(
    username: str,
    repo_name: str,
    token: Optional[str] = None,
    experiment_name: str = "fine-tuning",
) -> str:
    """
    Configura MLflow com DagsHub.
    
    Args:
        username: Username do DagsHub
        repo_name: Nome do repositório
        token: Token do DagsHub (opcional, usa env var se não fornecido)
        experiment_name: Nome do experimento
        
    Returns:
        URI de tracking do MLflow
    """
    try:
        import mlflow
        import dagshub
    except ImportError:
        logger.error("mlflow e/ou dagshub não instalados")
        raise
    
    # Token
    token = token or os.environ.get("DAGSHUB_TOKEN")
    if not token:
        logger.warning("DAGSHUB_TOKEN não definido. Algumas operações podem falhar.")
    
    # Inicializar DagsHub
    dagshub.init(repo_name=repo_name, repo_owner=username, mlflow=True)
    
    # Configurar MLflow
    tracking_uri = f"https://dagshub.com/{username}/{repo_name}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Credenciais
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    if token:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    
    # Criar/selecionar experimento
    mlflow.set_experiment(experiment_name)
    
    logger.info(f"MLflow configurado: {tracking_uri}")
    return tracking_uri


def validate_credentials(
    require_hf: bool = True,
    require_dagshub: bool = False,
) -> Dict[str, bool]:
    """
    Valida se as credenciais necessárias estão configuradas.
    
    Args:
        require_hf: Se requer token do Hugging Face
        require_dagshub: Se requer credenciais do DagsHub
        
    Returns:
        Dicionário com status de cada credencial
    """
    status = {}
    
    # Hugging Face
    hf_token = os.environ.get("HF_TOKEN")
    status["hf_token"] = bool(hf_token)
    
    if require_hf and not hf_token:
        logger.warning("HF_TOKEN não encontrado. Modelos como LLaMA requerem autenticação.")
    
    # DagsHub
    dagshub_token = os.environ.get("DAGSHUB_TOKEN")
    dagshub_user = os.environ.get("DAGSHUB_USERNAME")
    status["dagshub_token"] = bool(dagshub_token)
    status["dagshub_username"] = bool(dagshub_user)
    
    if require_dagshub:
        if not dagshub_token:
            logger.warning("DAGSHUB_TOKEN não encontrado.")
        if not dagshub_user:
            logger.warning("DAGSHUB_USERNAME não encontrado.")
    
    return status


class Timer:
    """Context manager para medir tempo de execução."""
    
    def __init__(self, name: str = "Operação", logger_func=None):
        """
        Inicializa o timer.
        
        Args:
            name: Nome da operação sendo medida
            logger_func: Função de log (padrão: logger.info)
        """
        self.name = name
        self.logger_func = logger_func or logger.info
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        self.logger_func(f"{self.name}: {self.format_time(self.elapsed)}")
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Formata tempo em formato legível."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{int(minutes)}m {secs:.1f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{int(hours)}h {int(minutes)}m"


def get_memory_usage() -> Dict[str, float]:
    """
    Retorna uso de memória atual.
    
    Returns:
        Dicionário com uso de memória (GPU e RAM)
    """
    import psutil
    
    memory = {
        "ram_used_gb": psutil.Process().memory_info().rss / (1024**3),
        "ram_percent": psutil.virtual_memory().percent,
    }
    
    if torch.cuda.is_available():
        memory["gpu_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
        memory["gpu_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
        memory["gpu_max_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
    
    return memory


def clear_gpu_memory() -> None:
    """Limpa memória da GPU."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("Memória GPU liberada")


def print_banner(text: str, char: str = "=", width: int = 60) -> None:
    """
    Imprime um banner formatado.
    
    Args:
        text: Texto do banner
        char: Caractere para a borda
        width: Largura total do banner
    """
    print(char * width)
    print(f"{text:^{width}}")
    print(char * width)


def format_number(num: int) -> str:
    """
    Formata número grande de forma legível.
    
    Args:
        num: Número a formatar
        
    Returns:
        String formatada (ex: 1.5B, 300M, 50K)
    """
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(num)


if __name__ == "__main__":
    # Testes básicos
    print_banner("PurrTune - Utils Test")
    
    env = get_execution_environment()
    print(f"Ambiente: {env}")
    
    gpu_available = check_gpu()
    if gpu_available:
        print_gpu_info()
    
    print(f"\nCredenciais: {validate_credentials(require_hf=False)}")
    
    with Timer("Teste de timer"):
        time.sleep(0.5)
    
    print("\n[OK] Utils funcionando!")
