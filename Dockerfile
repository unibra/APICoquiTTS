FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Instalar Python 3.9 e ferramentas básicas
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    curl \
    && ln -s /usr/bin/python3.9 /usr/bin/python \
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Instalar Python 3.9 e ferramentas básicas
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    curl \
    && ln -s /usr/bin/python3.9 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    gcc \
    g++ \
    make \
    libsndfile1 \
    espeak \
    espeak-data \
    libespeak1 \
    libespeak-dev \
    ffmpeg \
    git \
    git \
    && rm -rf /var/lib/apt/lists/*

# Definir diretório de trabalho
WORKDIR /app

# Copiar requirements e instalar dependências Python
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY app/ .

# Criar diretório para modelos (cache)
RUN mkdir -p /app/models

# Criar diretório para modelos (cache)
RUN mkdir -p /app/models

# Expor porta
EXPOSE 8888

# Comando para executar a aplicação
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"]