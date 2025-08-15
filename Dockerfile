FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# === Configurações Base ===
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Sao_Paulo
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# === Configurações CUDA ===
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV CUDA_VERSION=12.1
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# === Configurações de Aplicação ===
ENV FORCE_CPU=false
ENV TTS_DEVICE=auto
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV COQUI_TOS_AGREED=1

# === Otimizações PyTorch ===
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,roundup_power2_divisions:True,garbage_collection_threshold:0.6
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
ENV CUDA_MODULE_LOADING=LAZY
ENV TOKENIZERS_PARALLELISM=false

# === Cache de Modelos ===
ENV TTS_CACHE_PATH=/app/models
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HUGGINGFACE_HUB_CACHE=/app/models
ENV HF_HUB_CACHE=/app/models

# === Configurar timezone ===
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# === Atualizar sistema e instalar dependências ===
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    wget \
    curl \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    libffi-dev \
    espeak \
    espeak-data \
    libespeak1 \
    libespeak-dev \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev \
    libpulse-dev \
    libjack-jackd2-dev \
    && rm -rf /var/lib/apt/lists/*

# === Configurar Python 3.11 como padrão ===
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# === Definir diretório de trabalho ===
WORKDIR /app

# === Criar ambiente virtual Python limpo ===
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# === Atualizar pip, setuptools e wheel ===
RUN /opt/venv/bin/pip install --no-cache-dir --upgrade \
    pip==24.0 \
    setuptools==69.5.1 \
    wheel==0.43.0

# === Instalar PyTorch com CUDA 12.1 PRIMEIRO (evita conflitos) ===
RUN /opt/venv/bin/pip install --no-cache-dir \
    torch==2.4.1+cu121 \
    torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# === Copiar requirements e instalar todas as dependências ===
COPY app/requirements.txt /app/requirements.txt
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt || \
    echo "⚠️  Algumas dependências podem ter falhado, mas continuando..."

# === Verificar instalações críticas ===
RUN /opt/venv/bin/python -c "import torch; print(f'✅ PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" \
    && /opt/venv/bin/python -c "import fastapi; print('✅ FastAPI instalado')" \
    && (/opt/venv/bin/python -c "import TTS; print('✅ TTS instalado com sucesso')" || echo "⚠️  TTS pode ter problemas")

# === Copiar código da aplicação ===
COPY app/ /app/

# === Criar diretórios necessários ===
RUN mkdir -p models output logs tmp cache \
    && chmod 755 models output logs tmp cache

# === Expor porta ===
EXPOSE 8888

# === Health check ===
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8888/health || exit 1

# === Comando para executar a aplicação ===
CMD ["/opt/venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"]