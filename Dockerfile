FROM python:3.11-slim

# Instalar dependências básicas do sistema
RUN apt-get update && apt-get install -y \
    wget \
    curl \
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

# Instalar PyTorch com CUDA (se disponível)
RUN pip install --no-cache-dir \
    torch==2.4.1+cu121 \
    torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Instalar bibliotecas de áudio primeiro
RUN pip install --no-cache-dir \
    soundfile==0.12.1 \
    librosa==0.10.1 \
    numpy==1.24.4 \
    scipy==1.11.0

# Instalar Coqui TTS
RUN pip install --no-cache-dir TTS==0.22.0

# Instalar bibliotecas de monitoramento GPU
RUN pip install --no-cache-dir \
    nvidia-ml-py3==11.525.112 \
    gputil==1.4.0

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