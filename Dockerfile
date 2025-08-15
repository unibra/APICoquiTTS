FROM python:3.11-slim

# Instalar CUDA 12.4 toolkit e drivers para RTX 5090
RUN apt-get update && apt-get install -y \
    gnupg2 \
    software-properties-common && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" && \
    apt-get update

# Instalar dependências básicas do sistema
RUN apt-get update && apt-get install -y \
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
    cuda-toolkit-12-8 \
    cuda-runtime-12-8 \
    cuda-drivers-520 \
    && rm -rf /var/lib/apt/lists/*

# Configurar variáveis de ambiente NVIDIA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV CUDA_VERSION=12.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_REQUIRE_CUDA="cuda>=12.8"
ENV TORCH_CUDA_ARCH_LIST="5.0 6.0 7.0 7.5 8.0 8.6 9.0+PTX"

# Definir diretório de trabalho
WORKDIR /app

# Atualizar pip para versão mais recente
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copiar requirements primeiro (para cache de layers)
COPY app/requirements.txt .

# Instalar dependências Python do requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Instalar PyTorch nightly com CUDA 12.8 e suporte RTX 5090 (sm_120)
RUN pip install --no-cache-dir \
    torch==2.5.1+cu128 \
    torchvision==0.20.1+cu128 \
    torchaudio==2.5.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Instalar bibliotecas de áudio primeiro
RUN pip install --no-cache-dir \
    soundfile==0.12.1 \
    librosa==0.10.1 \
    numpy==1.24.4 \
    scipy==1.11.0

# Instalar Coqui TTS versão mais recente com melhor suporte CUDA
RUN pip install --no-cache-dir \
    git+https://github.com/coqui-ai/TTS.git@dev \
    --upgrade --force-reinstall

# Instalar bibliotecas de monitoramento GPU
RUN pip install --no-cache-dir \
    nvidia-ml-py3==7.352.0 \
    gputil==1.4.0 \
    pynvml==11.5.0

# Copiar código da aplicação
COPY app/ .

# Criar diretórios necessários com permissões corretas
RUN mkdir -p models output logs tmp cache && \
    chmod 755 models output logs tmp cache && \
    find . -type f -name "*.py" -exec chmod 644 {} \; && \
    find . -type f -name "*.sh" -exec chmod 755 {} \; && \
    find . -type f -name "*.json" -exec chmod 644 {} \; && \
    find . -type f -name "*.txt" -exec chmod 644 {} \; && \
    find . -type f -name "*.md" -exec chmod 644 {} \; && \
    find . -type d -exec chmod 755 {} \; && \
    chmod 755 /app

# Expor porta
EXPOSE 8888

# Comando para executar a aplicação
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"]