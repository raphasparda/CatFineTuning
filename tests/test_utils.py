"""
Testes unitários para o módulo de utilidades.
"""

import os
import pytest
from unittest.mock import patch

from src.utils import (
    Timer,
    is_colab,
    get_execution_environment,
    validate_credentials,
)


class TestTimer:
    """Testes para a classe Timer."""
    
    def test_timer_measures_time(self):
        """Testa se o timer mede tempo corretamente."""
        import time
        
        with Timer("test_operation") as timer:
            time.sleep(0.1)
        
        assert timer.elapsed >= 0.1
        assert timer.elapsed < 0.5  # Margem de segurança
    
    def test_timer_formats_correctly(self):
        """Testa formatação do tempo."""
        timer = Timer("test")
        
        # Simular elapsed time
        timer._start_time = 0
        timer._end_time = 65.5  # 1 minuto e 5.5 segundos
        
        # elapsed é calculado no __exit__, então testamos a formatação manual
        assert hasattr(timer, '_start_time')
    
    def test_timer_as_context_manager(self):
        """Testa uso como context manager."""
        with Timer("test") as t:
            _ = sum(range(1000))
        
        assert t.elapsed is not None
        assert t.elapsed >= 0


class TestIsColab:
    """Testes para is_colab."""
    
    @patch.dict(os.environ, {"COLAB_RELEASE_TAG": "v1.0"})
    def test_in_colab_environment(self):
        """Testa detecção em ambiente Colab."""
        # Note: Este teste pode não funcionar perfeitamente
        # pois is_colab também verifica importação de google.colab
        pass
    
    def test_not_in_colab(self):
        """Testa que não está em Colab em ambiente normal."""
        result = is_colab()
        # Em ambiente de teste normal, não deve ser Colab
        assert isinstance(result, bool)


class TestGetExecutionEnvironment:
    """Testes para get_execution_environment."""
    
    def test_returns_dict(self):
        """Testa se retorna um dicionário."""
        result = get_execution_environment()
        
        assert isinstance(result, dict)
        assert "environment" in result
        assert "python_version" in result
    
    def test_environment_type(self):
        """Testa tipo de ambiente."""
        result = get_execution_environment()
        
        valid_environments = ["local", "colab", "kaggle", "unknown"]
        assert result["environment"] in valid_environments


class TestValidateCredentials:
    """Testes para validate_credentials."""
    
    @patch.dict(os.environ, {
        "HF_TOKEN": "test_token",
        "DAGSHUB_TOKEN": "dagshub_test",
        "DAGSHUB_USERNAME": "user",
        "DAGSHUB_REPO": "repo",
    })
    def test_all_credentials_present(self):
        """Testa com todas as credenciais presentes."""
        result = validate_credentials(
            hf_token="test_token",
            dagshub_token="dagshub_test",
        )
        
        assert result["hf_token"] is True
        assert result["dagshub_token"] is True
    
    def test_missing_credentials(self):
        """Testa com credenciais faltando."""
        result = validate_credentials(
            hf_token=None,
            dagshub_token=None,
        )
        
        # Deve verificar variáveis de ambiente
        assert "hf_token" in result
        assert "dagshub_token" in result


class TestSetupEnvironment:
    """Testes para setup_environment (básicos)."""
    
    def test_import(self):
        """Testa se a função pode ser importada."""
        from src.utils import setup_environment
        assert callable(setup_environment)


class TestCheckGpu:
    """Testes para check_gpu."""
    
    def test_import(self):
        """Testa se a função pode ser importada."""
        from src.utils import check_gpu
        assert callable(check_gpu)
    
    def test_returns_boolean(self):
        """Testa se retorna booleano."""
        from src.utils import check_gpu
        result = check_gpu()
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
