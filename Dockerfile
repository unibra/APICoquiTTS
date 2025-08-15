FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Configurar variáveis de ambiente NVIDIA e CUDA 12.1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV CUDA_VERSION=12.1
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_REQUIRE_CUDA="cuda>=12.1"
ENV TORCH_CUDA_ARCH_LIST="5.0 6.0 7.0 7.5 8.0 8.6 9.0+PTX"

# Instalar Python 3.11 e dependências do sistema
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
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
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Configurar Python 3.11 como padrão
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Atualizar pip para versão mais recente
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Definir diretório de trabalho
WORKDIR /app

# Copiar requirements primeiro (para cache de layers)
COPY app/requirements.txt .

# Instalar dependências Python básicas
RUN pip install --no-cache-dir -r requirements.txt

# Instalar PyTorch com CUDA 12.1 (alinhado com outros serviços)
RUN pip install --no-cache-dir \
    torch==2.4.1+cu121 \
    torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Instalar bibliotecas de áudio
RUN pip install --no-cache-dir \
    soundfile==0.12.1 \
    librosa==0.10.1 \
    numpy==1.26.4 \
    scipy==1.12.0

# Instalar Coqui TTS - versão específica que funciona bem
RUN pip install --no-cache-dir \
    TTS==0.22.0 \
    --no-deps

# Reinstalar dependências específicas do TTS
RUN pip install --no-cache-dir \
    coqui-ai-tts \
    transformers>=4.33.0 \
    tokenizers>=0.13.3 \
    datasets>=2.14.0 \
    fsspec>=2023.9.2 \
    aiohttp \
    gdown \
    packaging \
    psutil \
    pyyaml \
    requests \
    tqdm \
    unicodedata2 \
    inflect \
    anyascii \
    pysbd \
    gruut[de,es,fr] \
    bangla \
    bnnumerizer \
    bnunicodenormalizer

# Instalar bibliotecas de monitoramento GPU
RUN pip install --no-cache-dir \
    nvidia-ml-py3==7.352.0 \
    gputil==1.4.0 \
    pynvml==11.5.0

# Copiar código da aplicação
COPY app/ .

# Criar diretórios necessários com permissões corretas
RUN mkdir -p models output logs tmp cache && \
    chmod 755 models output logs tmp cache

# Configurar variáveis de ambiente para otimização GPU
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,roundup_power2_divisions:True,garbage_collection_threshold:0.6
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
ENV CUDA_MODULE_LOADING=LAZY
ENV TORCH_COMPILE_DEBUG=0
ENV TORCH_LOGS=""

# Expor porta
EXPOSE 8888

# Comando para executar a aplicação
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"]