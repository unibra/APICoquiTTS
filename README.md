# Coqui TTS API Service com CUDA

Serviço de Text-to-Speech (TTS) usando Coqui TTS, FastAPI e Docker Compose com aceleração CUDA.

## Funcionalidades

- 🎤 Conversão de texto em áudio usando modelos Coqui TTS
- 🚀 Aceleração GPU com CUDA 12.1
- 🚀 API REST com FastAPI
- 🐳 Containerização com Docker
- 📊 Documentação automática da API
- 🔧 Configuração flexível de modelos e vozes
- 📈 Health checks e monitoramento
- ⚡ PyTorch 2.4.1 com CUDA 12.1
- 🔥 Suporte a Tensor Cores modernas

**Otimizações GPU:**
- **Base Python 3.11** - Com cuDNN 9.11.0 e CUDA 12.1
- **PyTorch 2.4.1+cu121** - Alinhado com outros serviços
- **Tensor Cores** - Habilitado com `allow_tf32=True`
- **Precisão mista** - `torch.set_float32_matmul_precision('high')`
- **Benchmark automático** - `torch.backends.cudnn.benchmark = True`
- **Gerenciamento de memória** - Cache otimizado e limpeza automática

**Recursos GPU:**
- Monitoramento em tempo real (uso, temperatura, memória)
- Detecção automática de capacidade computacional
- Logs detalhados das configurações GPU
- Runtime NVIDIA com isolamento adequado
- Cache inteligente de modelos HuggingFace
- Configurações otimizadas do PyTorch CUDA

**Performance:**
- PyTorch 2.4.1 com CUDA 12.1
- Bibliotecas otimizadas (nvidia-ml-py3, GPUtil)
- Cache inteligente de modelos
- Processamento paralelo otimizado

Para usar com aceleração GPU, certifique-se de ter:
1. NVIDIA Docker instalado (`nvidia-docker2`)
2. Drivers NVIDIA atualizados (535.86.10+)
3. Docker Compose 3.8+ com suporte GPU

## Estrutura do Projeto

```
.
├── app/
│   ├── main.py           # Aplicação FastAPI
│   └── requirements.txt  # Dependências Python
├── Dockerfile            # Imagem Docker
├── docker-compose.yml    # Orquestração de containers
└── README.md            # Este arquivo
```

## Como Usar

### Pré-requisitos

- Docker com runtime NVIDIA instalado
- GPU NVIDIA com CUDA support
- Driver NVIDIA 525.60.13 ou superior
- Docker Compose 3.8+

### 1. Construir e Executar com Docker Compose

```bash
# Construir e iniciar os serviços
docker-compose up -d

# Verificar logs
docker-compose logs -f tts-api

# Parar os serviços
docker-compose down
```

### 2. Acessar a API

- **API Docs**: http://localhost:8888/docs
- **Health Check**: http://localhost:8888/health
- **Modelos Disponíveis**: http://localhost:8888/models

### 3. Usar o Endpoint TTS

#### Exemplo com curl:

```bash
curl -X POST "http://localhost:8888/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Olá, este é um teste do serviço TTS!",
    "model_name": "tts_models/en/ljspeech/tacotron2-DDC"
  }' \
  --output audio.wav
```

#### Exemplo com Python:

```python
import requests

url = "http://localhost:8888/tts"
data = {
    "text": "Olá mundo! Este é um teste do Coqui TTS.",
    "model_name": "tts_models/en/ljspeech/tacotron2-DDC"
}

response = requests.post(url, json=data)

if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print("Áudio gerado com sucesso!")
else:
    print(f"Erro: {response.status_code}")
```

## Parâmetros da API

### POST /tts

- `text` (obrigatório): Texto a ser convertido em áudio
- `model_name` (opcional): Nome do modelo TTS
- `speaker` (opcional): Nome do speaker/voz
- `language` (opcional): Código do idioma

### Modelos Suportados

Para ver todos os modelos disponíveis, acesse: `GET /models`

Alguns modelos populares:
- `tts_models/en/ljspeech/tacotron2-DDC`
- `tts_models/pt/cv/vits`
- `tts_models/es/mai/tacotron2-DDC`

## Configuração

### Variáveis de Ambiente

- `TTS_CACHE_PATH`: Caminho para cache de modelos (padrão: `/app/models`)
- `PYTHONUNBUFFERED`: Para logs em tempo real

### Volumes

- `./models:/app/models`: Cache persistente de modelos TTS
- `./output:/app/output`: Diretório para arquivos de saída (opcional)

## Desenvolvimento Local

### Sem Docker:

```bash
# Instalar dependências
pip install -r app/requirements.txt

# Executar aplicação
cd app
python main.py
```

### Com Docker apenas:

```bash
# Construir imagem
docker build -t coqui-tts-api .

# Executar container
docker run -p 8888:8888 coqui-tts-api
```

## Troubleshooting

### Problemas Comuns:

1. **Modelo não carrega**: Verifique se há espaço suficiente em disco
2. **Erro de dependências**: Rebuild a imagem Docker
3. **Memória insuficiente**: Aumente os recursos do Docker

### Logs:

```bash
# Ver logs da API
docker-compose logs tts-api
```

## Produção

Para produção, considere:

- Usar um modelo TTS mais rápido
- Implementar cache de áudio
- Adicionar autenticação (pode ser feito via Cloudflare Access)
- Configurar rate limiting
- Usar HTTPS (via Cloudflared tunnel)
- Implementar métricas e monitoramento

## Licença

Este projeto usa Coqui TTS, que está sob licença MPL 2.0.