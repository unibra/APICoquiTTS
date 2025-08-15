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
import time
import numpy as np
import soundfile as sf

# === Configurar logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === Variáveis globais de controle ===
TTS = None
TTS_AVAILABLE = False
GPU_UTILS_AVAILABLE = False
tts_model = None
device_type = "cpu"
gpu_available = False

# === Importar GPUtil opcionalmente ===
try:
    import GPUtil
    GPU_UTILS_AVAILABLE = True
    logger.info("✅ GPUtil importado com sucesso")
except ImportError:
    logger.warning("⚠️  GPUtil não disponível - monitoramento GPU limitado")
    GPUtil = None

def setup_gpu_optimization():
    """
    Configurar otimizações GPU para CUDA 12.1
    Retorna True se GPU estiver funcionando, False para fallback CPU
    """
    global gpu_available, device_type
    
    try:
        logger.info("🔍 Verificando disponibilidade CUDA...")
        
        # Verificação básica de CUDA
        if not torch.cuda.is_available():
            logger.info("🖥️  CUDA não disponível no PyTorch")
            return False
            
        # Verificar se há GPUs disponíveis
        device_count = torch.cuda.device_count()
        if device_count == 0:
            logger.info("🖥️  Nenhuma GPU CUDA encontrada")
            return False
            
        logger.info(f"🎯 {device_count} GPU(s) CUDA detectada(s)")
        
        try:
            # Tentar obter informações da GPU (pode falhar com "stoi" error)
            gpu_info = torch.cuda.get_device_properties(0)
            logger.info(f"🎯 GPU: {gpu_info.name}")
            logger.info(f"💾 Memória GPU: {gpu_info.total_memory / 1024**3:.1f} GB")
            logger.info(f"⚡ Compute Capability: {gpu_info.major}.{gpu_info.minor}")
            
        except RuntimeError as e:
            if "stoi" in str(e):
                logger.warning("⚠️  Erro CUDA 'stoi' detectado - problema comum com runtime")
                logger.info("🔄 Tentando continuar com CUDA básico...")
            else:
                logger.error(f"❌ Erro CUDA específico: {e}")
                logger.warning("🔄 Fazendo fallback para CPU")
                return False
        
        # Testar operação básica CUDA
        try:
            logger.info("🧪 Testando operação CUDA...")
            test_tensor = torch.tensor([1.0]).cuda()
            result = test_tensor * 2
            result.cpu()
            del test_tensor, result
            torch.cuda.empty_cache()
            logger.info("✅ Teste CUDA bem-sucedido!")
            
        except Exception as cuda_test_error:
            logger.error(f"❌ Teste CUDA falhou: {cuda_test_error}")
            logger.warning("🔄 Fazendo fallback para CPU")
            return False
        
        # Aplicar otimizações
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        torch.cuda.empty_cache()
        
        logger.info("🚀 GPU configurada e funcionando!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro na configuração GPU: {e}")
        logger.warning("🔄 Fazendo fallback para CPU")
        return False

# === Configurar GPU na inicialização ===
gpu_available = setup_gpu_optimization()
device_type = "cuda" if gpu_available else "cpu"

logger.info("=" * 60)
logger.info("🎯 CONFIGURAÇÃO FINAL DO SISTEMA")
logger.info("=" * 60)
logger.info(f"🖥️  Device: {device_type.upper()}")
logger.info(f"🎵 TTS: Aguardando carregamento...")
if gpu_available:
    logger.info("🚀 Modo: GPU Acelerada (CUDA 12.1)")
else:
    logger.info("🔄 Modo: CPU Fallback")
logger.info("=" * 60)

# === Inicializar FastAPI ===
app = FastAPI(
    title="Coqui TTS API",
    description="Serviço de Text-to-Speech usando Coqui TTS com CUDA 12.1",
    version="2.0.0"
)

# === Modelos de Request ===
class TTSRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    text: str
    model_name: Optional[str] = None
    speaker: Optional[str] = None
    language: Optional[str] = "pt"
    use_gpu: Optional[bool] = True
    speed: Optional[float] = 1.0

@app.on_event("startup")
async def startup_event():
    """Inicializar o modelo TTS na inicialização"""
    global TTS, TTS_AVAILABLE, tts_model
    
    logger.info("🚀 Iniciando aplicação TTS...")
    
    # === Tentar importar TTS ===
    try:
        logger.info("📦 Tentando importar Coqui TTS...")
        from TTS.api import TTS as TTS_Class
        TTS = TTS_Class
        TTS_AVAILABLE = True
        logger.info("✅ Coqui TTS importado com sucesso!")
        
    except ImportError as e:
        logger.error(f"❌ Falha ao importar TTS: {e}")
        logger.error("💡 Solução: docker exec -it tts-api pip install TTS")
        TTS_AVAILABLE = False
        return
    
    except Exception as e:
        logger.error(f"❌ Erro inesperado ao importar TTS: {e}")
        TTS_AVAILABLE = False
        return
    
    # === Configurar licença ===
    os.environ['COQUI_TOS_AGREED'] = '1'
    
    # === Tentar carregar modelo ===
    try:
        logger.info(f"🎯 Carregando modelo TTS no device: {device_type}")
        
        # Lista de modelos para tentar
        models_to_try = [
            ("tts_models/pt/cv/vits", "Modelo VITS português (recomendado)"),
            ("tts_models/en/ljspeech/tacotron2-DDC", "Modelo Tacotron2 inglês"),
            ("tts_models/en/ljspeech/glow-tts", "Modelo GlowTTS inglês (backup)")
        ]
        
        for model_name, description in models_to_try:
            try:
                logger.info(f"📥 Tentando: {description}")
                
                if gpu_available:
                    temp_model = TTS(model_name=model_name, progress_bar=True, gpu=True)
                    logger.info("🚀 Carregando modelo na GPU...")
                else:
                    temp_model = TTS(model_name=model_name, progress_bar=True, gpu=False)
                    logger.info("🖥️  Carregando modelo na CPU...")
                
                # Teste básico
                test_text = "Olá" if "pt" in model_name else "Hello"
                test_wav = temp_model.tts(text=test_text)
                
                if test_wav is not None and len(test_wav) > 0:
                    logger.info(f"✅ Modelo funcionando! Audio: {len(test_wav)} samples")
                    tts_model = temp_model
                    logger.info(f"🎉 Modelo carregado: {model_name}")
                    break
                    
            except Exception as model_error:
                logger.error(f"❌ Falha em {model_name}: {model_error}")
                continue
        
        if tts_model is None:
            logger.error("❌ Nenhum modelo TTS pôde ser carregado!")
            return
        
        # === Informações finais ===
        model_name = getattr(tts_model, 'model_name', 'Unknown')
        speakers = getattr(tts_model, 'speakers', None)
        
        logger.info("🎉 TTS inicializado com sucesso!")
        logger.info(f"📝 Modelo: {model_name}")
        logger.info(f"🖥️  Device: {device_type}")
        
        if speakers and len(speakers) > 0:
            logger.info(f"🎤 Speakers: {len(speakers)} disponíveis")
        
        if gpu_available:
            try:
                memory_used = torch.cuda.memory_allocated(0) / 1024**3
                logger.info(f"🎯 GPU - Memória em uso: {memory_used:.1f}GB")
            except:
                logger.info("🎯 GPU ativa")
        else:
            cpu_count = psutil.cpu_count()
            memory_info = psutil.virtual_memory()
            logger.info(f"🖥️  CPU: {cpu_count} cores, {memory_info.available / 1024**3:.1f}GB RAM disponível")
            
    except Exception as e:
        logger.error(f"❌ Erro na inicialização: {e}", exc_info=True)
        logger.warning("⚠️  Aplicação iniciará sem modelo TTS")

@app.get("/")
async def root():
    """Endpoint raiz"""
    return {
        "message": "Coqui TTS API v2.0 - Refatorada e Otimizada",
        "docs": "/docs",
        "device": device_type,
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Verificação de saúde"""
    gpu_info = {}
    
    if gpu_available and GPU_UTILS_AVAILABLE and GPUtil:
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
            logger.warning(f"Erro ao obter info GPU: {e}")
    
    return {
        "status": "healthy",
        "tts_installed": TTS_AVAILABLE,
        "tts_model_loaded": tts_model is not None,
        "gpu_available": gpu_available,
        "device": device_type,
        "mode": "GPU Accelerated" if gpu_available else "CPU Fallback",
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_info": gpu_info
    }

@app.get("/models")
async def list_models():
    """Listar modelos disponíveis"""
    if not TTS_AVAILABLE:
        raise HTTPException(status_code=500, detail="TTS não disponível")
    
    try:
        temp_tts = TTS()
        models = temp_tts.list_models()
        return {"available_models": models or []}
    except Exception as e:
        logger.error(f"Erro ao listar modelos: {e}")
        fallback_models = [
            "tts_models/pt/cv/vits",
            "tts_models/en/ljspeech/tacotron2-DDC",
            "tts_models/en/ljspeech/glow-tts",
            "tts_models/multilingual/multi-dataset/xtts_v2"
        ]
        return {
            "available_models": fallback_models,
            "note": "Lista de fallback",
            "error": str(e)
        }

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Converter texto em áudio"""
    if not TTS_AVAILABLE or tts_model is None:
        raise HTTPException(status_code=500, detail="TTS não disponível")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Texto vazio")
    
    try:
        logger.info(f"🎵 Gerando áudio: {request.text[:50]}...")
        
        start_time = time.time()
        
        # Usar modelo global ou carregar específico
        current_tts = tts_model
        if request.model_name:
            logger.info(f"📦 Carregando modelo específico: {request.model_name}")
            use_gpu = gpu_available and request.use_gpu
            current_tts = TTS(model_name=request.model_name, gpu=use_gpu)
        
        # Gerar áudio
        tts_kwargs = {"text": request.text}
        
        # Adicionar parâmetros opcionais
        if request.speaker:
            tts_kwargs["speaker"] = request.speaker
        if request.language:
            tts_kwargs["language"] = request.language
        
        wav_data = current_tts.tts(**tts_kwargs)
        
        inference_time = time.time() - start_time
        logger.info(f"✅ Áudio gerado em {inference_time:.2f}s")
        
        # Converter para WAV
        if not isinstance(wav_data, np.ndarray):
            wav_data = np.array(wav_data)
        
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, wav_data, 22050, format='WAV')
        audio_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(audio_buffer.read()),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=tts_output.wav"}
        )
        
    except Exception as e:
        logger.error(f"❌ Erro ao gerar áudio: {e}")
        raise HTTPException(status_code=500, detail=f"Erro TTS: {str(e)}")

@app.post("/tts-simple")
async def simple_tts(text: str):
    """Endpoint TTS simplificado"""
    request = TTSRequest(text=text)
    return await text_to_speech(request)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)