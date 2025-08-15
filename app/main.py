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
    
    try:
        # Inicializar com modelo XTTS v2 para Português do Brasil
        default_model = "tts_models/multilingual/multi-dataset/xtts_v2"
        logger.info(f"Carregando modelo TTS: {default_model}")
        
        # Configurar device (GPU se disponível)
        device = "cuda" if gpu_available else "cpu"
        logger.info(f"Usando device: {device}")
        
        tts_model = TTS(model_name=default_model).to(device)
        logger.info("Modelo XTTS v2 para Português do Brasil carregado com sucesso!")
        
        if gpu_available:
            logger.info("Otimizações GPU ativadas!")
        else:
            logger.info("Usando CPU para processamento TTS")
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
        # Criar instância temporária para acessar lista de modelos
        temp_tts = TTS()
        models = temp_tts.list_models()
        logger.info(f"Encontrados {len(models) if models else 0} modelos")
        return {"available_models": models or []}
    except Exception as e:
        logger.error(f"Erro ao listar modelos: {e}", exc_info=True)
        # Fallback com modelos conhecidos - priorizando XTTS v2 e português brasileiro
        fallback_models = [
            "tts_models/multilingual/multi-dataset/xtts_v2",
            "tts_models/pt/cv/vits",
            "tts_models/en/ljspeech/tacotron2-DDC",
            "tts_models/en/ljspeech/glow-tts",
            "tts_models/pt/cv/tacotron2-DDC",
            "tts_models/fr/mai/tacotron2-DDC",
            "tts_models/es/mai/tacotron2-DDC",
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
        if request.model_name and request.model_name != "tts_models/multilingual/multi-dataset/xtts_v2":
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
        
        # Verificar se o modelo suporta múltiplos speakers
        try:
            # Tentar obter informações sobre speakers do modelo
            speakers = getattr(current_tts, 'speakers', None)
            is_multi_speaker = speakers is not None and len(speakers) > 0
            
            # Gerar áudio com parâmetros apropriados
            tts_kwargs = {"text": request.text}
            
            # Adicionar speaker apenas se o modelo suportar e foi especificado
            if is_multi_speaker and request.speaker:
                tts_kwargs["speaker"] = request.speaker
                logger.info(f"Usando speaker: {request.speaker}")
            
            # Adicionar language se especificado
            if request.language:
                tts_kwargs["language"] = request.language
                logger.info(f"Usando idioma: {request.language} (Português do Brasil para 'pt')")
            
            logger.info(f"Parâmetros TTS: {tts_kwargs}")
            wav_data = current_tts.tts(**tts_kwargs)
            
        except Exception as tts_error:
            logger.error(f"Erro específico do TTS: {tts_error}")
            # Fallback - tentar apenas com texto
            logger.info("Tentando fallback apenas com texto...")
            wav_data = current_tts.tts(text=request.text)
        
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

@app.post("/tts-clone")
async def voice_cloning_tts(
    text: str = Form(...),
    language: str = Form("pt"),
    use_gpu: bool = Form(True),
    speaker_audio: UploadFile = File(...)
):
    """
    🎭 Clone de Voz com XTTS v2
    
    Gera áudio usando a voz de referência fornecida (zero-shot voice cloning).
    
    - **text**: Texto a ser convertido em áudio
    - **language**: Código do idioma (pt para Português do Brasil)
    - **use_gpu**: Usar GPU para processamento (se disponível)
    - **speaker_audio**: Arquivo de áudio com a voz de referência (WAV, MP3, etc.)
    
    📋 Requisitos para o áudio de referência:
    - Duração: 6-12 segundos (ideal)
    - Qualidade: Limpo, sem ruído
    - Formato: WAV, MP3, M4A, FLAC
    - Conteúdo: Apenas uma pessoa falando
    """
    logger.info(f"🎭 Iniciando clone de voz - Texto: '{text[:50]}...', Idioma: {language}")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Texto não pode estar vazio")
    
    if TTS is None:
        raise HTTPException(status_code=500, detail="Coqui TTS não está disponível")
    
    # Verificar se temos um modelo carregado
    if tts_model is None:
        raise HTTPException(status_code=500, detail="Modelo TTS não está carregado. Tente novamente em alguns segundos.")
    
    # Validar tipo de arquivo
    logger.info(f"📁 Arquivo recebido - Nome: {speaker_audio.filename}, Tipo: {speaker_audio.content_type}, Tamanho: {speaker_audio.size if hasattr(speaker_audio, 'size') else 'Desconhecido'}")
    
    # Verificar extensão do arquivo se o content_type não estiver disponível
    valid_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']
    if speaker_audio.filename:
        file_ext = Path(speaker_audio.filename).suffix.lower()
        if not (speaker_audio.content_type and speaker_audio.content_type.startswith('audio/')) and file_ext not in valid_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Arquivo deve ser de áudio. Extensões suportadas: {', '.join(valid_extensions)}"
            )
    
    # Criar arquivo temporário para o áudio de referência
    temp_audio_path = None
    try:
        logger.info("📁 Criando arquivo temporário para áudio de referência...")
        
        # Criar arquivo temporário
        file_suffix = Path(speaker_audio.filename).suffix if speaker_audio.filename else '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
            logger.info(f"📁 Copiando conteúdo para: {temp_file.name}")
            # Copiar conteúdo do upload para o arquivo temporário
            shutil.copyfileobj(speaker_audio.file, temp_file)
            temp_audio_path = temp_file.name
        
        # Verificar se o arquivo foi criado e tem conteúdo
        if not Path(temp_audio_path).exists():
            raise Exception("Falha ao criar arquivo temporário")
        
        file_size = Path(temp_audio_path).stat().st_size
        logger.info(f"✅ Arquivo temporário criado: {temp_audio_path} (tamanho: {file_size} bytes)")
        
        if file_size == 0:
            raise Exception("Arquivo de áudio está vazio")
        
        logger.info(f"🎭 Gerando clone de voz para: '{text[:50]}...'")
        
        # Verificar se o modelo suporta clonagem de voz
        model_name = getattr(tts_model, 'model_name', 'Unknown')
        logger.info(f"🤖 Usando modelo: {model_name}")
        
        # Verificar se é XTTS v2 ou modelo compatível
        if 'xtts' not in model_name.lower():
            logger.warning(f"⚠️  Modelo atual ({model_name}) pode não suportar clonagem de voz otimamente")
        
        # Medir tempo de inferência
        start_time = time.time()
        
        logger.info(f"🔧 Parâmetros TTS:")
        logger.info(f"   - Texto: '{text[:50]}...'")
        logger.info(f"   - Idioma: {language}")
        logger.info(f"   - Arquivo de referência: {temp_audio_path}")
        logger.info(f"   - Device: {'cuda' if gpu_available else 'cpu'}")
        
        try:
            # XTTS v2 suporta clonagem com speaker_wav
            logger.info("🎵 Executando síntese TTS com clonagem...")
            wav_data = tts_model.tts(
                text=text,
                speaker_wav=temp_audio_path,
                language=language
            )
            logger.info("✅ Síntese TTS concluída com sucesso!")
            
        except Exception as tts_error:
            logger.error(f"❌ Erro na síntese TTS: {type(tts_error).__name__}: {str(tts_error)}")
            # Tentar fallback sem especificar language
            logger.info("🔄 Tentando fallback sem parâmetro language...")
            try:
                wav_data = tts_model.tts(
                    text=text,
                    speaker_wav=temp_audio_path
                )
                logger.info("✅ Fallback bem-sucedido!")
            except Exception as fallback_error:
                logger.error(f"❌ Fallback também falhou: {type(fallback_error).__name__}: {str(fallback_error)}")
                raise Exception(f"Erro na síntese TTS: {str(tts_error)}. Fallback: {str(fallback_error)}")
        
        inference_time = time.time() - start_time
        logger.info(f"⏱️  Áudio gerado em {inference_time:.2f} segundos")
        
        # Verificar se temos dados de áudio
        if wav_data is None:
            raise Exception("Modelo retornou dados vazios")
        
        logger.info(f"📊 Dados de áudio: tipo={type(wav_data)}, shape={getattr(wav_data, 'shape', 'N/A')}")
        
        # Criar buffer em memória para o áudio
        logger.info("💾 Convertendo áudio para formato WAV...")
        audio_buffer = io.BytesIO()
        
        # Converter para bytes e escrever no buffer
        import soundfile as sf
        import numpy as np
        
        # Converter para numpy array se necessário
        if not isinstance(wav_data, np.ndarray):
            logger.info("🔄 Convertendo dados para numpy array...")
            wav_data = np.array(wav_data)
        
        # Verificar se temos dados válidos
        if len(wav_data) == 0:
            raise Exception("Dados de áudio estão vazios após conversão")
        
        logger.info(f"📊 Array final: shape={wav_data.shape}, dtype={wav_data.dtype}")
        
        # Escrever no buffer como WAV
        sample_rate = 22050  # Taxa de amostragem padrão do XTTS v2
        sf.write(audio_buffer, wav_data, sample_rate, format='WAV')
        audio_buffer.seek(0)
        
        # Verificar tamanho do buffer
        buffer_size = audio_buffer.getbuffer().nbytes
        logger.info(f"💾 Buffer de áudio criado: {buffer_size} bytes")
        
        if buffer_size == 0:
            raise Exception("Buffer de áudio está vazio")
        
        logger.info("🎉 Clone de voz gerado com sucesso! 🎭")
        
        # Retornar como streaming response
        return StreamingResponse(
            io.BytesIO(audio_buffer.read()),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=voice_clone_output.wav"}
        )
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"❌ Erro ao gerar clone de voz: {error_msg}")
        logger.error(f"📍 Detalhes do erro:", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao gerar clone de voz: {error_msg}")
    
    finally:
        # Limpar arquivo temporário
        if temp_audio_path and Path(temp_audio_path).exists():
            try:
                Path(temp_audio_path).unlink()
                logger.info(f"🗑️  Arquivo temporário removido: {temp_audio_path}")
            except Exception as e:
                logger.warning(f"⚠️  Erro ao remover arquivo temporário: {e}")

@app.get("/clone-info")
async def voice_clone_info():
    """
    📖 Informações sobre a funcionalidade de clonagem de voz
    """
    return {
        "feature": "Voice Cloning com XTTS v2",
        "description": "Clone qualquer voz usando apenas 6-12 segundos de áudio de referência",
        "endpoint": "/tts-clone",
        "method": "POST",
        "supported_languages": [
            "pt - Português (Brasil)",
            "en - English",
            "es - Español", 
            "fr - Français",
            "de - Deutsch",
            "it - Italiano",
            "ja - 日本語",
            "ko - 한국어",
            "zh - 中文",
            "ar - العربية",
            "tr - Türkçe",
            "pl - Polski",
            "nl - Nederlands",
            "cs - Čeština",
            "ru - Русский",
            "hu - Magyar",
            "hi - हिन्दी"
        ],
        "audio_requirements": {
            "duration": "6-12 segundos (ideal)",
            "quality": "Limpo, sem ruído de fundo",
            "formats": ["WAV", "MP3", "M4A", "FLAC"],
            "content": "Apenas uma pessoa falando",
            "language_match": "Preferencialmente no mesmo idioma de saída"
        },
        "tips": [
            "Use áudios com boa qualidade para melhores resultados",
            "Evite música ou ruído de fundo no áudio de referência",
            "6-12 segundos é o tempo ideal - nem muito curto, nem muito longo",
            "A voz clonada funcionará melhor no mesmo idioma do áudio original"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)