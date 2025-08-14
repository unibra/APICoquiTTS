from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import logging
from typing import Optional
import uvicorn
import torch
import os
import psutil
import threading
import time

try:
    from TTS.api import TTS
    import GPUtil
except ImportError:
    TTS = None
    GPUtil = None

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações de otimização para RTX 5090
def setup_gpu_optimization():
    """Configurar otimizações específicas para RTX 5090"""
    if torch.cuda.is_available():
        # Configurar para RTX 5090 (Ada Lovelace architecture)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Otimizações de memória
        torch.cuda.empty_cache()
        
        # Configurações específicas do ambiente
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Para performance
        
        # Configurar precision otimizada
        torch.set_float32_matmul_precision('high')
        torch.cuda.empty_cache()
        
        # Log das configurações GPU
        gpu_info = torch.cuda.get_device_properties(0)
        logger.info(f"GPU detectada: {gpu_info.name}")
        logger.info(f"Memória GPU: {gpu_info.total_memory / 1024**3:.1f} GB")
        logger.info(f"Compute Capability: {gpu_info.major}.{gpu_info.minor}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        
        return True
    return False

# Configurar GPU na inicialização
gpu_available = setup_gpu_optimization()

app = FastAPI(
    title="Coqui TTS API",
    description="Serviço de Text-to-Speech usando Coqui TTS otimizado para NVIDIA RTX 5090",
    version="1.0.0"
)

# Modelo de request
class TTSRequest(BaseModel):
    text: str
    model_name: Optional[str] = "tts_models/en/ljspeech/tacotron2-DDC"
    speaker: Optional[str] = None
    language: Optional[str] = "en"
    use_gpu: Optional[bool] = True
    speed: Optional[float] = 1.0

# Variável global para armazenar o modelo TTS
tts_model = None

@app.on_event("startup")
async def startup_event():
    """Inicializar o modelo TTS na inicialização da aplicação"""
    global tts_model
    
    if TTS is None:
        logger.error("Coqui TTS não está instalado. Instale com: pip install TTS")
        return
    
    try:
        # Inicializar com modelo padrão
        default_model = "tts_models/en/ljspeech/tacotron2-DDC"
        logger.info(f"Carregando modelo TTS: {default_model}")
        
        # Configurar device (GPU se disponível)
        device = "cuda" if gpu_available else "cpu"
        logger.info(f"Usando device: {device}")
        
        tts_model = TTS(model_name=default_model).to(device)
        logger.info("Modelo TTS carregado com sucesso!")
        
        if gpu_available:
            logger.info("Otimizações RTX 5090 ativadas!")
    except Exception as e:
        logger.error(f"Erro ao carregar modelo TTS: {e}")

@app.get("/")
async def root():
    """Endpoint raiz com informações sobre a API"""
    return {
        "message": "Coqui TTS API está funcionando!",
        "docs": "/docs",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Verificação de saúde da aplicação"""
    gpu_info = {}
    if gpu_available and GPUtil:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_info = {
                    "name": gpu.name,
                    "memory_used": f"{gpu.memoryUsed}MB",
                    "memory_total": f"{gpu.memoryTotal}MB",
                    "gpu_load": f"{gpu.load*100:.1f}%",
                    "temperature": f"{gpu.temperature}°C"
                }
        except:
            pass
    
    gpu_info = {}
    if gpu_available and GPUtil:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_info = {
                    "name": gpu.name,
                    "memory_used": f"{gpu.memoryUsed}MB",
                    "memory_total": f"{gpu.memoryTotal}MB",
                    "gpu_load": f"{gpu.load*100:.1f}%",
                    "temperature": f"{gpu.temperature}°C"
                }
        except:
            pass
    
    return {
        "status": "healthy",
        "tts_available": tts_model is not None,
        "gpu_available": gpu_available,
        "device": "cuda" if gpu_available else "cpu",
        "gpu_info": gpu_info
    }

@app.get("/models")
async def list_models():
    """Listar modelos TTS disponíveis"""
    if TTS is None:
        raise HTTPException(status_code=500, detail="Coqui TTS não está disponível")
    
    try:
        models = TTS.list_models()
        return {"available_models": models}
    except Exception as e:
        logger.error(f"Erro ao listar modelos: {e}")
        raise HTTPException(status_code=500, detail="Erro ao listar modelos")

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Converter texto em áudio usando Coqui TTS
    
    - **text**: Texto a ser convertido em áudio
    - **model_name**: Nome do modelo TTS a ser usado (opcional)
    - **speaker**: Nome do speaker/voz (opcional, depende do modelo)
    - **language**: Código do idioma (opcional)
    """
    if TTS is None:
        raise HTTPException(status_code=500, detail="Coqui TTS não está disponível")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Texto não pode estar vazio")
    
    try:
        # Usar modelo global ou carregar um novo se especificado
        current_tts = tts_model
        if request.model_name and request.model_name != "tts_models/en/ljspeech/tacotron2-DDC":
            logger.info(f"Carregando modelo específico: {request.model_name}")
            device = "cuda" if (gpu_available and request.use_gpu) else "cpu"
            current_tts = TTS(model_name=request.model_name).to(device)
        
        if current_tts is None:
            raise HTTPException(status_code=500, detail="Modelo TTS não está carregado")
        
        # Gerar áudio
        device = "cuda" if (gpu_available and request.use_gpu) else "cpu"
        logger.info(f"Gerando áudio para texto: {request.text[:50]}... (device: {device})")
        
        # Criar buffer em memória para o áudio
        audio_buffer = io.BytesIO()
        
        # Parâmetros para TTS
        tts_kwargs = {}
        if request.speaker:
            tts_kwargs["speaker"] = request.speaker
        if request.language:
            tts_kwargs["language"] = request.language
        if request.speed != 1.0:
            tts_kwargs["speed"] = request.speed
        
        # Medir tempo de inferência
        start_time = time.time()
        
        # Converter para bytes e escrever no buffer
        import soundfile as sf
        sf.write(audio_buffer, wav_data, 22050, format='WAV')
        audio_buffer.seek(0)
        
        logger.info("Áudio gerado com sucesso!")
        
        # Retornar como streaming response
        return StreamingResponse(
            io.BytesIO(audio_buffer.read()),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=tts_output.wav"}
        )
    except Exception as e:
        logger.error(f"Erro ao gerar áudio: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar áudio: {str(e)}")

@app.post("/tts-simple")
async def simple_text_to_speech(text: str):
    """
    Versão simplificada do endpoint TTS - apenas recebe texto como parâmetro
    """
    request = TTSRequest(text=text)
    return await text_to_speech(request)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)