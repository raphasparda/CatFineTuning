import json

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "accelerator": "GPU"
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Pipeline de Fine-Tuning de LLMs com QLoRA\n",
                "\n",
                "## MLOps com Google Colab + DagsHub/MLflow\n",
                "\n",
                "| Componente | Tecnologia | Custo |\n",
                "|------------|------------|-------|\n",
                "| **Compute** | Google Colab T4 GPU (16GB VRAM) | Gratuito |\n",
                "| **Modelo** | LLaMA 3.2 (3B Instruct) | Open Source |\n",
                "| **Dataset** | Guanaco 1K | Open Source |\n",
                "| **Otimizacao** | QLoRA + bitsandbytes (4-bit) | Open Source |\n",
                "| **Tracking** | DagsHub + MLflow | Free Tier |\n",
                "\n",
                "> **IMPORTANTE**: LLaMA 3 requer autenticacao no Hugging Face:\n",
                "> 1. Criar conta em huggingface.co\n",
                "> 2. Aceitar os termos em https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct\n",
                "> 3. Gerar um token de acesso"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Instalacao de Dependencias"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "!pip install -q torch transformers>=4.40.0 datasets>=2.14.0 accelerate>=0.24.0 peft>=0.6.0 bitsandbytes>=0.41.0 trl>=0.7.0 mlflow>=2.8.0 dagshub>=0.3.0 huggingface_hub>=0.19.0 sentencepiece protobuf\n",
                "print(\"[OK] Dependencias instaladas!\")"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Verificacao de GPU e Imports"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "import torch\n",
                "import os\n",
                "import warnings\n",
                "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments\n",
                "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType\n",
                "from trl import SFTTrainer\n",
                "from datasets import Dataset, load_dataset\n",
                "import mlflow\n",
                "import dagshub\n",
                "from huggingface_hub import login\n",
                "\n",
                "warnings.filterwarnings(\"ignore\")\n",
                "os.environ[\"TOKENIZERS_PARALLELISM\"] = \"false\"\n",
                "\n",
                "if torch.cuda.is_available():\n",
                "    gpu_name = torch.cuda.get_device_name(0)\n",
                "    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
                "    print(f\"[OK] GPU: {gpu_name} ({gpu_memory:.1f} GB)\")\n",
                "else:\n",
                "    raise RuntimeError(\"GPU nao disponivel! Va em Runtime -> Change runtime type -> GPU\")"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Autenticacao Hugging Face (Obrigatorio para LLaMA 3)\n",
                "\n",
                "**Passos:**\n",
                "1. Acesse https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct\n",
                "2. Aceite os termos de uso\n",
                "3. Va em Settings -> Access Tokens -> New Token\n",
                "4. Cole o token abaixo"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Opcao 1: Cole seu token diretamente (menos seguro)\n",
                "HF_TOKEN = \"seu_token_aqui\"\n",
                "\n",
                "# Opcao 2: Use secrets do Colab (mais seguro)\n",
                "# from google.colab import userdata\n",
                "# HF_TOKEN = userdata.get(\"HF_TOKEN\")\n",
                "\n",
                "login(token=HF_TOKEN)\n",
                "print(\"[OK] Autenticado no Hugging Face!\")"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. Configuracoes do Pipeline"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# ============================================================================\n",
                "# CONFIGURACOES PRINCIPAIS\n",
                "# ============================================================================\n",
                "\n",
                "# Modelo LLaMA 3\n",
                "MODEL_NAME = \"meta-llama/Llama-3.2-3B-Instruct\"\n",
                "\n",
                "# Dataset do Hugging Face\n",
                "DATASET_NAME = \"mlabonne/guanaco-llama2-1k\"\n",
                "\n",
                "# Configuracao de quantizacao 4-bit\n",
                "QUANTIZATION_CONFIG = {\n",
                "    \"load_in_4bit\": True,\n",
                "    \"bnb_4bit_compute_dtype\": torch.float16,\n",
                "    \"bnb_4bit_quant_type\": \"nf4\",\n",
                "    \"bnb_4bit_use_double_quant\": True,\n",
                "}\n",
                "\n",
                "# Configuracao LoRA\n",
                "LORA_CONFIG = {\n",
                "    \"r\": 16,\n",
                "    \"lora_alpha\": 32,\n",
                "    \"lora_dropout\": 0.05,\n",
                "    \"bias\": \"none\",\n",
                "    \"task_type\": \"CAUSAL_LM\",\n",
                "    \"target_modules\": [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"],\n",
                "}\n",
                "\n",
                "# Configuracao de treinamento\n",
                "TRAINING_CONFIG = {\n",
                "    \"num_train_epochs\": 1,\n",
                "    \"per_device_train_batch_size\": 2,\n",
                "    \"gradient_accumulation_steps\": 4,\n",
                "    \"learning_rate\": 2e-4,\n",
                "    \"warmup_ratio\": 0.03,\n",
                "    \"lr_scheduler_type\": \"cosine\",\n",
                "    \"max_seq_length\": 1024,\n",
                "    \"logging_steps\": 10,\n",
                "    \"save_steps\": 50,\n",
                "}\n",
                "\n",
                "OUTPUT_DIR = \"./outputs\"\n",
                "\n",
                "print(f\"[OK] Modelo: {MODEL_NAME}\")\n",
                "print(f\"[OK] Dataset: {DATASET_NAME}\")"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 5. Configuracao do MLflow com DagsHub (Opcional)"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Preencha suas credenciais do DagsHub\n",
                "DAGSHUB_USERNAME = \"seu_username\"\n",
                "DAGSHUB_REPO = \"seu_repositorio\"\n",
                "DAGSHUB_TOKEN = \"seu_token\"\n",
                "\n",
                "def setup_mlflow():\n",
                "    dagshub.init(repo_name=DAGSHUB_REPO, repo_owner=DAGSHUB_USERNAME, mlflow=True)\n",
                "    tracking_uri = f\"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow\"\n",
                "    mlflow.set_tracking_uri(tracking_uri)\n",
                "    os.environ[\"MLFLOW_TRACKING_USERNAME\"] = DAGSHUB_USERNAME\n",
                "    os.environ[\"MLFLOW_TRACKING_PASSWORD\"] = DAGSHUB_TOKEN\n",
                "    mlflow.set_experiment(\"llama3-fine-tuning\")\n",
                "    print(f\"[OK] MLflow: {tracking_uri}\")\n",
                "    return tracking_uri\n",
                "\n",
                "# Descomente para ativar MLflow\n",
                "# tracking_uri = setup_mlflow()"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 6. Carregamento do Modelo com Quantizacao 4-bit"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "print(\"=\"*60)\n",
                "print(f\"Carregando modelo: {MODEL_NAME}\")\n",
                "print(\"=\"*60)\n",
                "\n",
                "bnb_config = BitsAndBytesConfig(\n",
                "    load_in_4bit=QUANTIZATION_CONFIG[\"load_in_4bit\"],\n",
                "    bnb_4bit_compute_dtype=QUANTIZATION_CONFIG[\"bnb_4bit_compute_dtype\"],\n",
                "    bnb_4bit_quant_type=QUANTIZATION_CONFIG[\"bnb_4bit_quant_type\"],\n",
                "    bnb_4bit_use_double_quant=QUANTIZATION_CONFIG[\"bnb_4bit_use_double_quant\"],\n",
                ")\n",
                "\n",
                "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)\n",
                "if tokenizer.pad_token is None:\n",
                "    tokenizer.pad_token = tokenizer.eos_token\n",
                "\n",
                "model = AutoModelForCausalLM.from_pretrained(\n",
                "    MODEL_NAME,\n",
                "    quantization_config=bnb_config,\n",
                "    device_map=\"auto\",\n",
                "    trust_remote_code=True,\n",
                ")\n",
                "\n",
                "model = prepare_model_for_kbit_training(model)\n",
                "memory_gb = model.get_memory_footprint() / (1024**3)\n",
                "print(f\"\\n[OK] Modelo carregado! Memoria: {memory_gb:.2f} GB\")"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 7. Carregamento do Dataset Guanaco"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "print(f\"Carregando dataset: {DATASET_NAME}\")\n",
                "\n",
                "dataset = load_dataset(DATASET_NAME, split=\"train\")\n",
                "\n",
                "print(f\"\\n[OK] Dataset carregado!\")\n",
                "print(f\"[STATS] Total de amostras: {len(dataset)}\")\n",
                "print(f\"[STATS] Colunas: {dataset.column_names}\")\n",
                "print(f\"\\n[EXEMPLO]:\\n{dataset[0]['text'][:500]}...\")"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 8. Aplicacao do LoRA"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "lora_config = LoraConfig(\n",
                "    r=LORA_CONFIG[\"r\"],\n",
                "    lora_alpha=LORA_CONFIG[\"lora_alpha\"],\n",
                "    lora_dropout=LORA_CONFIG[\"lora_dropout\"],\n",
                "    bias=LORA_CONFIG[\"bias\"],\n",
                "    task_type=TaskType.CAUSAL_LM,\n",
                "    target_modules=LORA_CONFIG[\"target_modules\"],\n",
                ")\n",
                "\n",
                "model = get_peft_model(model, lora_config)\n",
                "\n",
                "trainable, total = 0, 0\n",
                "for _, param in model.named_parameters():\n",
                "    total += param.numel()\n",
                "    if param.requires_grad:\n",
                "        trainable += param.numel()\n",
                "\n",
                "print(f\"[OK] LoRA aplicado!\")\n",
                "print(f\"[STATS] Parametros treinaveis: {trainable:,} ({100*trainable/total:.2f}%)\")\n",
                "print(f\"[STATS] Parametros totais: {total:,}\")"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 9. Treinamento com SFTTrainer"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "training_args = TrainingArguments(\n",
                "    output_dir=OUTPUT_DIR,\n",
                "    num_train_epochs=TRAINING_CONFIG[\"num_train_epochs\"],\n",
                "    per_device_train_batch_size=TRAINING_CONFIG[\"per_device_train_batch_size\"],\n",
                "    gradient_accumulation_steps=TRAINING_CONFIG[\"gradient_accumulation_steps\"],\n",
                "    learning_rate=TRAINING_CONFIG[\"learning_rate\"],\n",
                "    warmup_ratio=TRAINING_CONFIG[\"warmup_ratio\"],\n",
                "    lr_scheduler_type=TRAINING_CONFIG[\"lr_scheduler_type\"],\n",
                "    logging_steps=TRAINING_CONFIG[\"logging_steps\"],\n",
                "    save_steps=TRAINING_CONFIG[\"save_steps\"],\n",
                "    save_total_limit=2,\n",
                "    fp16=True,\n",
                "    gradient_checkpointing=True,\n",
                "    optim=\"paged_adamw_32bit\",\n",
                "    report_to=\"none\",\n",
                ")\n",
                "\n",
                "trainer = SFTTrainer(\n",
                "    model=model,\n",
                "    train_dataset=dataset,\n",
                "    args=training_args,\n",
                "    tokenizer=tokenizer,\n",
                "    dataset_text_field=\"text\",\n",
                "    max_seq_length=TRAINING_CONFIG[\"max_seq_length\"],\n",
                "    packing=False,\n",
                ")\n",
                "\n",
                "print(\"[OK] Trainer configurado!\")"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "print(\"=\"*60)\n",
                "print(\"INICIANDO TREINAMENTO\")\n",
                "print(\"=\"*60)\n",
                "\n",
                "trainer.train()\n",
                "\n",
                "print(\"\\n[OK] Treinamento concluido!\")"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 10. Salvamento do Modelo"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "model_path = f\"{OUTPUT_DIR}/llama3-finetuned\"\n",
                "trainer.save_model(model_path)\n",
                "tokenizer.save_pretrained(model_path)\n",
                "\n",
                "print(f\"[OK] Modelo salvo em: {model_path}\")"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 11. Teste de Inferencia"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "def gerar_resposta(prompt, max_tokens=256):\n",
                "    messages = [{\"role\": \"user\", \"content\": prompt}]\n",
                "    \n",
                "    input_ids = tokenizer.apply_chat_template(\n",
                "        messages,\n",
                "        add_generation_prompt=True,\n",
                "        return_tensors=\"pt\"\n",
                "    ).to(model.device)\n",
                "    \n",
                "    outputs = model.generate(\n",
                "        input_ids,\n",
                "        max_new_tokens=max_tokens,\n",
                "        do_sample=True,\n",
                "        temperature=0.7,\n",
                "        top_p=0.9,\n",
                "        pad_token_id=tokenizer.eos_token_id,\n",
                "    )\n",
                "    \n",
                "    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)\n",
                "    return response\n",
                "\n",
                "# Teste\n",
                "print(\"=\"*60)\n",
                "print(\"TESTE DE INFERENCIA\")\n",
                "print(\"=\"*60)\n",
                "\n",
                "perguntas = [\n",
                "    \"O que e Machine Learning?\",\n",
                "    \"Explique o conceito de fine-tuning em LLMs.\",\n",
                "    \"Quais sao as vantagens do QLoRA?\",\n",
                "]\n",
                "\n",
                "for pergunta in perguntas:\n",
                "    print(f\"\\n[PERGUNTA] {pergunta}\")\n",
                "    resposta = gerar_resposta(pergunta)\n",
                "    print(f\"[RESPOSTA] {resposta}\")"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 12. Conclusao\n",
                "\n",
                "Parabens! Voce treinou o LLaMA 3.2 3B com sucesso usando QLoRA.\n",
                "\n",
                "### Proximos passos:\n",
                "- Experimente com mais epochs\n",
                "- Teste diferentes datasets\n",
                "- Ajuste os hiperparametros\n",
                "- Faca push do modelo para o Hugging Face Hub"
            ]
        }
    ]
}

with open('notebooks/fine_tuning_pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print('[OK] Notebook atualizado para LLaMA 3!')
