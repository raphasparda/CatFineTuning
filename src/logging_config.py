"""
Configuração de logging estruturado para o pipeline de fine-tuning.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# Cores ANSI para terminal
class Colors:
    """Códigos de cores ANSI para formatação de terminal."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


class ColoredFormatter(logging.Formatter):
    """Formatter personalizado com cores para console."""
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BOLD + Colors.RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        # Adicionar cor ao nível de log
        color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
        record.levelname = f"{color}{record.levelname}{Colors.RESET}"
        
        # Adicionar cor ao nome do logger
        record.name = f"{Colors.BLUE}{record.name}{Colors.RESET}"
        
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    colored: bool = True,
) -> logging.Logger:
    """
    Configura o sistema de logging.
    
    Args:
        level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Nome do arquivo de log (opcional)
        log_dir: Diretório para salvar logs
        colored: Se True, usa cores no console
        
    Returns:
        Logger configurado
    """
    # Determinar nível numérico
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Criar logger raiz do projeto
    logger = logging.getLogger("purrtune")
    logger.setLevel(numeric_level)
    
    # Limpar handlers existentes
    logger.handlers.clear()
    
    # Formato base
    console_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    file_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if colored and sys.stdout.isatty():
        console_formatter = ColoredFormatter(console_format, datefmt=date_format)
    else:
        console_formatter = logging.Formatter(console_format, datefmt=date_format)
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Handler para arquivo (se especificado)
    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_path / log_file,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)  # Arquivo sempre captura tudo
        file_formatter = logging.Formatter(file_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Obtém um logger filho do logger principal.
    
    Args:
        name: Nome do módulo/componente
        
    Returns:
        Logger configurado
    """
    return logging.getLogger(f"purrtune.{name}")


def create_training_log_file() -> str:
    """
    Cria um nome de arquivo de log único para a sessão de treinamento.
    
    Returns:
        Nome do arquivo de log
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"training_{timestamp}.log"


# Logger padrão para uso rápido
class LoggerMixin:
    """Mixin para adicionar logging a classes."""
    
    @property
    def logger(self) -> logging.Logger:
        """Retorna o logger para a classe."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


# Inicialização padrão
_default_logger: Optional[logging.Logger] = None


def init_default_logger(level: str = "INFO", log_to_file: bool = False) -> logging.Logger:
    """
    Inicializa o logger padrão do projeto.
    
    Args:
        level: Nível de log
        log_to_file: Se True, também salva em arquivo
        
    Returns:
        Logger configurado
    """
    global _default_logger
    
    log_file = create_training_log_file() if log_to_file else None
    _default_logger = setup_logging(level=level, log_file=log_file)
    
    return _default_logger


def get_default_logger() -> logging.Logger:
    """
    Obtém o logger padrão, inicializando se necessário.
    
    Returns:
        Logger padrão
    """
    global _default_logger
    
    if _default_logger is None:
        _default_logger = init_default_logger()
    
    return _default_logger
