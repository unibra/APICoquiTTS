FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Configurar timezone e evitar prompts interativos
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Sao_Paulo
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Configurar variáveis CUDA
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV CUDA_VERSION=12.1
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Variáveis para fallback CPU
ENV FORCE_CPU=false
ENV TTS_DEVICE=auto

# Configurar timezone primeiro
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Atualizar sistema e instalar dependências essenciais
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

# Configurar Python 3.11 como padrão
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Definir diretório de trabalho
WORKDIR /app

# Criar ambiente virtual Python limpo
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Atualizar pip, setuptools e wheel no ambiente virtual
RUN /opt/venv/bin/pip install --no-cache-dir --upgrade \
    pip==24.0 \
    setuptools==69.5.1 \
    wheel==0.43.0

# Instalar PyTorch com CUDA 12.1 PRIMEIRO (evita conflitos)
RUN /opt/venv/bin/pip install --no-cache-dir \
    torch==2.4.1+cu121 \
    torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Instalar bibliotecas de áudio e processamento
RUN /opt/venv/bin/pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.11.4 \
    soundfile==0.12.1 \
    librosa==0.10.1 \
    resampy==0.4.2

# Instalar dependências do FastAPI
RUN /opt/venv/bin/pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    python-multipart==0.0.6

# Instalar bibliotecas de monitoramento (com fallback)
RUN /opt/venv/bin/pip install --no-cache-dir \
    psutil==5.9.6 \
    && /opt/venv/bin/pip install --no-cache-dir --ignore-installed \
    nvidia-ml-py3==7.352.0 \
    gputil==1.4.0 \
    pynvml==11.5.0 \
    || echo "GPU monitoring libs failed, continuing..."

# Instalar TTS com flags para evitar conflitos
RUN /opt/venv/bin/pip install --no-cache-dir --force-reinstall --no-deps \
    TTS==0.22.0 \
    && /opt/venv/bin/pip install --no-cache-dir \
    transformers>=4.33.0 \
    tokenizers>=0.13.3 \
    datasets>=2.14.0 \
    fsspec>=2023.9.2 \
    aiohttp \
    gdown \
    packaging \
    pyyaml \
    requests \
    tqdm \
    inflect \
    anyascii \
    pysbd \
    jieba \
    pypinyin \
    mecab-python3 \
    unidic-lite \
    bangla \
    bnnumerizer \
    bnunicodenormalizer \
    || echo "Some TTS dependencies failed, core should work"

# Instalar gruut separadamente (pode falhar mas não é crítico)
RUN /opt/venv/bin/pip install --no-cache-dir \
    gruut[de,es,fr] \
    || echo "Gruut failed, continuing without it"

# Copiar requirements e instalar dependências adicionais
COPY app/requirements.txt /app/requirements.txt
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt \
    || echo "Some requirements failed, checking core dependencies"

# Verificar instalações críticas
RUN /opt/venv/bin/python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" \
    && /opt/venv/bin/python -c "import TTS; print('TTS installed successfully')" \
    && /opt/venv/bin/python -c "import fastapi; print('FastAPI installed successfully')"

# Copiar código da aplicação
COPY app/ /app/

# Criar diretórios necessários
RUN mkdir -p models output logs tmp cache \
    && chmod 755 models output logs tmp cache

# Configurar variáveis de ambiente para otimização
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV COQUI_TOS_AGREED=1

# Variáveis de otimização PyTorch
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,roundup_power2_divisions:True
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
ENV CUDA_MODULE_LOADING=LAZY

# Cache de modelos
ENV TTS_CACHE_PATH=/app/models
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HUGGINGFACE_HUB_CACHE=/app/models

# Expor porta
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8888/health || exit 1

# Comando para executar a aplicação
CMD ["/opt/venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"]