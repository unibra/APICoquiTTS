FROM python:3.11-slim

# Instalar dependências básicas primeiro
RUN apt-get update && apt-get install -y \
    wget \
    gnupg2 \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Instalar cuDNN 9.11.0 seguindo as instruções oficiais
RUN wget https://developer.download.nvidia.com/compute/cudnn/9.11.0/local_installers/cudnn-local-repo-ubuntu2204-9.11.0_1.0-1_amd64.deb \
    && dpkg -i cudnn-local-repo-ubuntu2204-9.11.0_1.0-1_amd64.deb \
    && cp /var/cudnn-local-repo-ubuntu2204-9.11.0/cudnn-*-keyring.gpg /usr/share/keyrings/ \
    && apt-get update \
    && apt-get -y install cudnn-cuda-12 \
    && rm cudnn-local-repo-ubuntu2204-9.11.0_1.0-1_amd64.deb

# Instalar dependências do sistema para TTS
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libsndfile1 \
    espeak \
    espeak-data \
    libespeak1 \
    libespeak-dev \
    ffmpeg \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Configurar variáveis de ambiente NVIDIA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Definir diretório de trabalho
WORKDIR /app

# Atualizar pip para versão mais recente
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copiar requirements primeiro (para cache de layers)
COPY app/requirements.txt .

# Instalar dependências Python do requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Instalar PyTorch com CUDA 12.1
RUN pip install --no-cache-dir \
    torch==2.4.1+cu121 \
    torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Instalar bibliotecas de áudio e TTS
RUN pip install --no-cache-dir \
    soundfile==0.12.1 \
    librosa==0.10.1 \
    numpy==1.24.4 \
    scipy==1.11.0

# Instalar Coqui TTS
RUN pip install --no-cache-dir TTS==0.22.0

# Instalar bibliotecas de monitoramento GPU
RUN pip install --no-cache-dir \
    nvidia-ml-py3==12.560.30 \
    gputil==1.4.0 \
    numba==0.58.1

# Copiar código da aplicação
COPY app/ .

# Criar diretórios necessários com permissões corretas
RUN mkdir -p models output logs tmp cache && \
    chmod 755 models output logs tmp cache && \
    find . -type f -name "*.py" -exec chmod 644 {} \; && \
    find . -type d -exec chmod 755 {} \; && \
    chmod 755 /app

# Configurar cache de modelos
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HUGGINGFACE_HUB_CACHE=/app/models
ENV HF_HUB_CACHE=/app/models
ENV TOKENIZERS_PARALLELISM=false

# Expor porta
EXPOSE 8888

# Comando para executar a aplicação
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"]