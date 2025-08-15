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
        try:
            # Obter informações da GPU
            gpu_info = torch.cuda.get_device_properties(0)
            compute_capability = f"sm_{gpu_info.major}{gpu_info.minor}"
            
            logger.info(f"GPU detectada: {gpu_info.name}")
            logger.info(f"Memória GPU: {gpu_info.total_memory / 1024**3:.1f} GB")
            logger.info(f"Compute Capability: {gpu_info.major}.{gpu_info.minor} ({compute_capability})")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            
            # Verificar se a arquitetura é suportada pelo PyTorch
            supported_archs = torch.cuda.get_arch_list()
            logger.info(f"Arquiteturas CUDA suportadas pelo PyTorch: {supported_archs}")
            
            # Verificar se a capacidade computacional está nas arquiteturas suportadas
            arch_supported = any(compute_capability in arch for arch in supported_archs)
            
            if not arch_supported:
                logger.warning(f"⚠️  Arquitetura {compute_capability} não está explicitamente suportada")
                logger.warning("🔄 Fazendo fallback para CPU para evitar erros CUDA")
                return False
            
            # Testar uma operação simples na GPU para verificar compatibilidade real
            try:
                test_tensor = torch.tensor([1.0]).cuda()
                result = test_tensor * 2
                result.cpu()  # Mover de volta para CPU
                del test_tensor, result
                torch.cuda.empty_cache()
                logger.info("✅ Teste CUDA bem-sucedido!")
            except Exception as cuda_test_error:
                logger.error(f"❌ Teste CUDA falhou: {cuda_test_error}")
                logger.warning("🔄 Fazendo fallback para CPU")
                return False
            
            # Configurar otimizações GPU se tudo estiver OK
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Configurações específicas do ambiente
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Para performance
            
            # Configurar precision otimizada
            torch.set_float32_matmul_precision('high')
            torch.cuda.empty_cache()
            
            logger.info("🚀 Otimizações GPU ativadas com sucesso!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao configurar GPU: {e}")
            logger.warning("🔄 Fazendo fallback para CPU")
            return False
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
    model_config = {"protected_namespaces": ()}
    
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
        except Exception as e:
            logger.warning(f"Erro ao obter informações da GPU: {e}")
    
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
        logger.info("Tentando listar modelos TTS disponíveis...")
        models = TTS.list_models()
        logger.info(f"Encontrados {len(models) if models else 0} modelos")
        return {"available_models": models or []}
    except Exception as e:
        logger.error(f"Erro ao listar modelos: {e}", exc_info=True)
        # Fallback com modelos conhecidos
        fallback_models = [
            "tts_models/en/ljspeech/tacotron2-DDC",
            "tts_models/en/ljspeech/glow-tts",
            "tts_models/en/ljspeech/speedy-speech",
            "tts_models/pt/cv/vits",
            "tts_models/es/mai/tacotron2-DDC",
            "tts_models/fr/mai/tacotron2-DDC",
            "tts_models/de/mai/tacotron2-DDC"
        ]
        logger.info(f"Retornando lista de modelos conhecidos como fallback: {len(fallback_models)} modelos")
        return {
            "available_models": fallback_models,
            "note": "Lista de fallback - alguns modelos podem não estar disponíveis",
            "error": str(e)
        }

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
        
        # Medir tempo de inferência
        start_time = time.time()
        
        # Gerar áudio usando TTS
        wav_data = current_tts.tts(text=request.text, speaker=request.speaker, language=request.language)
        
        inference_time = time.time() - start_time
        logger.info(f"Áudio gerado em {inference_time:.2f} segundos")
        
        # Criar buffer em memória para o áudio
        audio_buffer = io.BytesIO()
        
        # Converter para bytes e escrever no buffer
        import soundfile as sf
        import numpy as np
        
        # Converter para numpy array se necessário
        if not isinstance(wav_data, np.ndarray):
            wav_data = np.array(wav_data)
        
        # Escrever no buffer como WAV
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