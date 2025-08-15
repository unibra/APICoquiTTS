from fastapi import FastAPI, HTTPException
from fastapi import File, UploadFile, Form
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
import tempfile
import shutil
from pathlib import Path

try:
    from TTS.api import TTS
    import GPUtil
except ImportError:
    TTS = None
    GPUtil = None

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações de otimização para CUDA
def setup_gpu_optimization():
    """
    Configurar otimizações GPU para CUDA 12.1 (PyTorch 2.4.1+cu121)
    Retorna True se GPU estiver funcionando, False para fallback CPU
    """
    if torch.cuda.is_available():
        try:
            logger.info("🔍 Verificando disponibilidade CUDA...")
            
            # Obter informações da GPU
            gpu_info = torch.cuda.get_device_properties(0)
            compute_capability = f"sm_{gpu_info.major}{gpu_info.minor}"
            
            logger.info(f"🎯 GPU detectada: {gpu_info.name}")
            logger.info(f"💾 Memória GPU: {gpu_info.total_memory / 1024**3:.1f} GB")
            logger.info(f"⚡ Compute Capability: {gpu_info.major}.{gpu_info.minor} ({compute_capability})")
            logger.info(f"PyTorch: {torch.__version__} | CUDA Runtime: {torch.version.cuda}")
            
            # Verificar se a arquitetura é suportada pelo PyTorch
            supported_archs = torch.cuda.get_arch_list()
            logger.info(f"🏗️  Arquiteturas CUDA suportadas: {supported_archs}")
            
            # Configurações para GPUs modernas
            if gpu_info.major >= 8:  # RTX 3000+ e superiores
                logger.info(f"🚀 GPU moderna {gpu_info.name} detectada! Aplicando otimizações...")
                
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
                torch.set_float32_matmul_precision('high')
                logger.info("⚡ Tensor Cores habilitados")
            else:
                logger.info("🖥️  Aplicando configurações padrão GPU")
            
            # Testar uma operação simples na GPU
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
                logger.warning("🔄 CUDA não funcional, fazendo fallback para CPU")
                return False
            
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"📊 Memória GPU - Reservada: {memory_reserved:.1f}GB, Alocada: {memory_allocated:.1f}GB")
            
            logger.info("🚀 GPU configurada e funcionando!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao configurar GPU: {e}", exc_info=True)
            logger.warning("🔄 GPU indisponível, fazendo fallback para CPU")
            return False
    else:
        logger.info("🖥️  CUDA não disponível no sistema")
        logger.info("🔄 Usando processamento CPU")
    return False

# Configurar GPU na inicialização
gpu_available = setup_gpu_optimization()
device_type = "cuda" if gpu_available else "cpu"

logger.info("=" * 60)
logger.info("🎯 CONFIGURAÇÃO FINAL DO SISTEMA")
logger.info("=" * 60)
logger.info(f"🖥️  Device: {device_type.upper()}")
logger.info(f"🎵 TTS: Coqui TTS com PyTorch {torch.__version__}")
if gpu_available:
    logger.info("🚀 Modo: GPU Acelerada (CUDA 12.1)")
else:
    logger.info("🔄 Modo: CPU Fallback")
logger.info("=" * 60)

app = FastAPI(
    title="Coqui TTS API",
    description="Serviço de Text-to-Speech usando Coqui TTS com CUDA 12.1 (PyTorch 2.4.1+cu121)",
    version="1.0.0"
)

# Modelo de request
class TTSRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    text: str
    model_name: Optional[str] = "tts_models/multilingual/multi-dataset/xtts_v2"
    speaker: Optional[str] = None
    language: Optional[str] = "pt"
    use_gpu: Optional[bool] = True
    speed: Optional[float] = 1.0

# Modelo para clonagem de voz
class VoiceCloneRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    text: str
    language: Optional[str] = "pt"
    use_gpu: Optional[bool] = True

# Variável global para armazenar o modelo TTS
tts_model = None

@app.on_event("startup")
async def startup_event():
    """Inicializar o modelo TTS na inicialização da aplicação"""
    global tts_model
    
    if TTS is None:
        logger.error("Coqui TTS não está instalado. Instale com: pip install TTS")
        return
    
    # Configurar variáveis de ambiente para aceitar licença automaticamente
    os.environ['COQUI_TOS_AGREED'] = '1'
    
    try:
        logger.info("🚀 Inicializando modelo TTS...")
        logger.info(f"🎯 Carregando modelo no device: {device_type}")
        
        # Lista de modelos para tentar (em ordem de preferência)
        models_to_try = [
            ("tts_models/pt/cv/vits", "Modelo VITS português (recomendado)"),
            ("tts_models/en/ljspeech/tacotron2-DDC", "Modelo Tacotron2 inglês (estável)"),
            ("tts_models/en/ljspeech/glow-tts", "Modelo GlowTTS inglês (backup)")
        ]
        
        tts_model = None
        
        for model_name, description in models_to_try:
            try:
                logger.info(f"📥 Tentando carregar: {description}")
                logger.info(f"⏳ Modelo: {model_name}")
                
                # Configurar modelo com device apropriado
                if gpu_available:
                    temp_model = TTS(model_name=model_name, progress_bar=True, gpu=True).to(device_type)
                    logger.info("🚀 Carregando modelo na GPU...")
                else:
                    temp_model = TTS(model_name=model_name, progress_bar=True, gpu=False)
                    logger.info("🖥️  Carregando modelo na CPU...")
                
                # Testar o modelo com uma frase simples
                test_text = "Olá" if "pt" in model_name else "Hello"
                
                try:
                    # Verificar se é multi-speaker
                    speakers = getattr(temp_model, 'speakers', None)
                    is_multi_speaker = speakers is not None and len(speakers) > 0
                    
                    logger.info(f"🔍 Multi-speaker: {is_multi_speaker}")
                    if is_multi_speaker:
                        logger.info(f"🎤 Speakers disponíveis: {speakers[:5]}...")
                    
                    # Teste básico
                    if is_multi_speaker and speakers:
                        test_speaker = speakers[0]
                        logger.info(f"🎤 Testando com speaker: {test_speaker}")
                        test_wav = temp_model.tts(text=test_text, speaker=test_speaker)
                    else:
                        test_wav = temp_model.tts(text=test_text)
                    
                    if test_wav is not None and len(test_wav) > 0:
                        logger.info(f"✅ Modelo funcionando! Tamanho do áudio: {len(test_wav)}")
                        tts_model = temp_model
                        break
                    else:
                        logger.warning("⚠️  Modelo não gerou áudio válido")
                        
                except Exception as test_error:
                    logger.error(f"❌ Teste do modelo falhou: {test_error}")
                    continue
                    
            except Exception as model_error:
                logger.error(f"❌ Falha ao carregar {model_name}: {model_error}")
                continue
        
        if tts_model is None:
            logger.error("❌ Nenhum modelo TTS pôde ser carregado!")
            return
            
        # Obter informações do modelo carregado
        model_name = getattr(tts_model, 'model_name', 'Unknown')
        speakers = getattr(tts_model, 'speakers', None)
        is_multi_speaker = speakers is not None and len(speakers) > 0
        
        logger.info(f"🎉 Modelo TTS carregado com sucesso!")
        logger.info(f"📝 Modelo: {model_name}")
        logger.info(f"🖥️  Device: {device_type}")
        logger.info(f"🎤 Multi-speaker: {is_multi_speaker}")
        
        # Log específico para GPU
        if gpu_available:
            try:
                memory_used = torch.cuda.memory_allocated(0) / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"🎯 GPU - Memória usada: {memory_used:.1f}GB/{memory_total:.1f}GB")
            except:
                logger.info("🎯 GPU ativa (informações de memória indisponíveis)")
        else:
            # Informações do sistema para CPU
            cpu_count = psutil.cpu_count()
            memory_info = psutil.virtual_memory()
            logger.info(f"🖥️  CPU: {cpu_count} cores disponíveis")
            logger.info(f"💾 RAM: {memory_info.available / 1024**3:.1f}GB/{memory_info.total / 1024**3:.1f}GB disponível")
            
        if speakers and len(speakers) > 0:
            logger.info(f"🔊 Total de speakers: {len(speakers)}")
            logger.info(f"🎵 Speakers disponíveis: {speakers[:10]}")
            
        try:
            # Verificar capacidades do modelo
            languages = getattr(tts_model, 'languages', None)
            if languages:
                logger.info(f"🌐 Idiomas suportados: {languages}")
        except:
            pass
            
        if gpu_available:
            logger.info("🚀 Usando GPU para processamento TTS acelerado!")
        else:
            logger.info("🖥️  Usando CPU para processamento TTS (modo compatibilidade)")
        
    except Exception as e:
        logger.error(f"❌ Erro crítico na inicialização: {e}", exc_info=True)
        logger.warning("⚠️  Serviço iniciará sem modelo TTS carregado")

@app.get("/")
async def root():
    """Endpoint raiz com informações sobre a API"""
    return {
        "message": "Coqui TTS API está funcionando!",
        "docs": "/docs",
        "version": "1.0.0",
        "device": device_type
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
        "device": device_type,
        "mode": "GPU Accelerated" if gpu_available else "CPU Fallback",
        "gpu_info": gpu_info
    }

@app.get("/models")
async def list_models():
    """Listar modelos TTS disponíveis"""
    if TTS is None:
        raise HTTPException(status_code=500, detail="Coqui TTS não está disponível")
    
    try:
        logger.info("Tentando listar modelos TTS disponíveis...")
        temp_tts = TTS()
        models = temp_tts.list_models()
        logger.info(f"Encontrados {len(models) if models else 0} modelos")
        return {"available_models": models or []}
    except Exception as e:
        logger.error(f"Erro ao listar modelos: {e}", exc_info=True)
        fallback_models = [
            "tts_models/multilingual/multi-dataset/xtts_v2",
            "tts_models/pt/cv/vits",
            "tts_models/en/ljspeech/tacotron2-DDC",
            "tts_models/en/ljspeech/glow-tts"
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
    """
    if TTS is None:
        raise HTTPException(status_code=500, detail="Coqui TTS não está disponível")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Texto não pode estar vazio")
    
    try:
        # Usar modelo global ou carregar um novo se especificado
        current_tts = tts_model
        if request.model_name and request.model_name != "tts_models/multilingual/multi-dataset/xtts_v2":
            logger.info(f"Carregando modelo específico: {request.model_name}")
            use_gpu = gpu_available and request.use_gpu
            if use_gpu:
                current_tts = TTS(model_name=request.model_name, gpu=True).to(device_type)
                logger.info("🚀 Modelo específico carregado na GPU")
            else:
                current_tts = TTS(model_name=request.model_name, gpu=False)
                logger.info("🖥️  Modelo específico carregado na CPU")
        
        if current_tts is None:
            raise HTTPException(status_code=500, detail="Modelo TTS não está carregado")
        
        # Gerar áudio
        use_gpu = gpu_available and request.use_gpu
        actual_device = device_type if use_gpu else "cpu"
        logger.info(f"Gerando áudio para texto: {request.text[:50]}... (device: {actual_device})")
        
        start_time = time.time()
        
        # Verificar se o modelo suporta múltiplos speakers
        try:
            speakers = getattr(current_tts, 'speakers', None)
            is_multi_speaker = speakers is not None and len(speakers) > 0
            
            tts_kwargs = {"text": request.text}
            
            if is_multi_speaker and request.speaker:
                tts_kwargs["speaker"] = request.speaker
                logger.info(f"Usando speaker: {request.speaker}")
            
            if request.language:
                tts_kwargs["language"] = request.language
                logger.info(f"Usando idioma: {request.language}")
            
            logger.info(f"Parâmetros TTS: {tts_kwargs}")
            wav_data = current_tts.tts(**tts_kwargs)
            
        except Exception as tts_error:
            logger.error(f"Erro específico do TTS: {tts_error}")
            logger.info("Tentando fallback apenas com texto...")
            wav_data = current_tts.tts(text=request.text)
        
        inference_time = time.time() - start_time
        logger.info(f"Áudio gerado em {inference_time:.2f} segundos")
        
        # Criar buffer em memória para o áudio
        audio_buffer = io.BytesIO()
        
        # Converter para bytes e escrever no buffer
        import soundfile as sf
        import numpy as np
        
        if not isinstance(wav_data, np.ndarray):
            wav_data = np.array(wav_data)
        
        sf.write(audio_buffer, wav_data, 22050, format='WAV')
        audio_buffer.seek(0)
        
        logger.info("Áudio gerado com sucesso!")
        
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
    Versão simplificada do endpoint TTS
    """
    request = TTSRequest(text=text)
    return await text_to_speech(request)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)