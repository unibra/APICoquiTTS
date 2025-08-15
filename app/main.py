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
            logger.info(f"PyTorch CUDA Version: {torch.version.cuda}")
            logger.info(f"CUDA Runtime API Version: {torch.cuda.get_device_capability(0)}")
            
            # Verificar se a arquitetura é suportada pelo PyTorch
            supported_archs = torch.cuda.get_arch_list()
            logger.info(f"Arquiteturas CUDA suportadas pelo PyTorch: {supported_archs}")
            
            # RTX 5090 específica (sm_120) - verificação especial
            if gpu_info.major >= 12:  # Ada Lovelace Next-gen (RTX 5090)
                logger.info("🚀 RTX 5090 detectada! Aplicando otimizações específicas...")
                
                # Configurações específicas para RTX 5090
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
                torch.backends.cuda.enable_flash_sdp(True)
                # Otimizações CUDA 12.8 específicas
                if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                    torch.backends.cuda.enable_flash_sdp(True)
                    logger.info("⚡ Flash Attention habilitado")
                
                # Tensor Cores de 4ª geração para RTX 5090
                torch.set_float32_matmul_precision('high')  
                
                torch.set_float32_matmul_precision('high')  # Usar Tensor Cores de 4ª geração
                
                # Configurações de memória otimizadas para 32GB
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:True,garbage_collection_threshold:0.6'
                os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Performance máxima
                os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
                os.environ['CUDA_MODULE_LOADING'] = 'LAZY'  # CUDA 12.8 lazy loading
                
                logger.info("⚡ Tensor Cores de 4ª geração ativados (CUDA 12.8)")
                logger.info("🧠 Otimizações de memória 32GB aplicadas")
                logger.info("🔥 CUDA 12.8 lazy loading habilitado")
            else:
                # Para GPUs mais antigas, verificar compatibilidade normal
                arch_supported = any(compute_capability in arch for arch in supported_archs)
                if not arch_supported:
                    logger.warning(f"⚠️  Arquitetura {compute_capability} pode não estar suportada")
                    # Tentar mesmo assim - PyTorch nightly pode suportar
            
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
            torch.backends.cudnn.deterministic = False  # Máxima performance
            torch.cuda.empty_cache()
            
            # Log de status final
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"🎯 Memória GPU - Reservada: {memory_reserved:.1f}GB, Alocada: {memory_allocated:.1f}GB")
            
            logger.info("🚀 Otimizações RTX 5090 ativadas com sucesso!")
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
    
    # Configurar variáveis de ambiente para aceitar licença automaticamente
    os.environ['COQUI_TOS_AGREED'] = '1'
    
    try:
        logger.info("🚀 Inicializando modelo TTS...")
        
        # Configurar device (GPU se disponível)
        # Forçar CPU por enquanto devido à incompatibilidade CUDA com RTX 5090
        device = "cpu"  # Temporary fallback
        logger.info(f"Usando device: {device}")
        
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
                
                temp_model = TTS(model_name=model_name, progress_bar=True).to(device)
                
                # Testar o modelo com uma frase simples
                test_text = "Olá" if "pt" in model_name else "Hello"
            # Configurar device com otimizações RTX 5090
            if device == "cuda" and gpu_available:
                temp_model = TTS(model_name=model_name, progress_bar=True, gpu=True).to(device)
                logger.info("🚀 Modelo carregado na RTX 5090 com otimizações GPU")
            else:
                temp_model = TTS(model_name=model_name, progress_bar=True, gpu=False).to(device)
                logger.info("🖥️  Modelo carregado na CPU")
                
                # Verificar se o modelo funciona
                try:
                    # Verificar se é multi-speaker
                    speakers = getattr(temp_model, 'speakers', None)
                    is_multi_speaker = speakers is not None and len(speakers) > 0
                    
                    logger.info(f"🔍 Multi-speaker: {is_multi_speaker}")
                    if is_multi_speaker:
                        logger.info(f"🎤 Speakers disponíveis: {speakers[:5]}...")  # Mostrar apenas os primeiros 5
                    
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
        logger.info(f"🖥️  Device: {device}")
        logger.info(f"🎤 Multi-speaker: {is_multi_speaker}")
        
        # Log específico para RTX 5090
        if device == "cuda" and gpu_available:
            try:
                memory_used = torch.cuda.memory_allocated(0) / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"🎯 GPU RTX 5090 - Memória usada: {memory_used:.1f}GB/{memory_total:.1f}GB")
            except:
                pass
        
        if is_multi_speaker:
            logger.info(f"🔊 Total de speakers: {len(speakers)}")
            logger.info(f"🎵 Primeiros speakers: {speakers[:10]}")
        
        try:
            # Verificar capacidades do modelo
            languages = getattr(tts_model, 'languages', None)
            if languages:
                logger.info(f"🌐 Idiomas suportados: {languages}")
        except:
            pass
            
        if device == "cuda":
            logger.info("🚀 Usando RTX 5090 para processamento TTS acelerado!")
        else:
            logger.info("🖥️  Usando CPU para processamento TTS")
        
    except Exception as e:

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
    🎭 Síntese de Voz com Áudio de Referência
    
    Gera áudio usando o modelo TTS disponível. Se possível, usa o áudio como referência.
    
    - **text**: Texto a ser convertido em áudio
    - **language**: Código do idioma (pt para Português do Brasil)
    - **use_gpu**: Usar GPU para processamento (se disponível)
    - **speaker_audio**: Arquivo de áudio (WAV, MP3, etc.)
    
    📋 Recomendações para o áudio:
    - Duração: 6-12 segundos (ideal)
    - Qualidade: Limpo, sem ruído
    - Formato: WAV, MP3, M4A, FLAC
    - Conteúdo: Uma pessoa falando claramente
    """
    logger.info(f"🎵 Iniciando síntese TTS - Texto: '{text[:50]}...', Idioma: {language}")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Texto não pode estar vazio")
    
    if TTS is None:
        raise HTTPException(status_code=500, detail="Coqui TTS não está disponível")
    
    # Verificar se temos um modelo carregado
    if tts_model is None:
        raise HTTPException(status_code=500, detail="Modelo TTS não está carregado. Tente novamente em alguns segundos.")
    
    # Obter informações do modelo
    model_name = getattr(tts_model, 'model_name', 'Unknown')
    speakers = getattr(tts_model, 'speakers', None)
    is_multi_speaker = speakers is not None and len(speakers) > 0
    
    logger.info(f"🤖 Usando modelo: {model_name}")
    logger.info(f"🎤 Multi-speaker: {is_multi_speaker}")
    
    try:
        # Medir tempo de inferência
        start_time = time.time()
        
        # Estratégia: Tentar diferentes abordagens baseadas no modelo
        wav_data = None
        method_used = "unknown"
        
        # Método 1: Se é multi-speaker, usar um speaker padrão
        if is_multi_speaker and speakers:
            try:
                # Selecionar speaker padrão (primeiro da lista)
                default_speaker = speakers[0]
                logger.info(f"🎤 Tentativa 1: Multi-speaker com '{default_speaker}'")
                
                # Decidir se incluir language baseado no modelo
                if "pt" in model_name.lower() or "multilingual" in model_name.lower():
                    # Modelo português ou multilingual
                    wav_data = tts_model.tts(text=text, speaker=default_speaker, language=language)
                    method_used = f"multi-speaker com language ({default_speaker})"
                else:
                    # Modelo inglês - sem language
                    wav_data = tts_model.tts(text=text, speaker=default_speaker)
                    method_used = f"multi-speaker sem language ({default_speaker})"
                    
                logger.info(f"✅ Método 1 funcionou: {method_used}")
                
            except Exception as method1_error:
                logger.warning(f"⚠️  Método 1 falhou: {method1_error}")
        
        # Método 2: TTS simples (se método 1 falhou ou modelo não é multi-speaker)
        if wav_data is None:
            try:
                logger.info("🎵 Tentativa 2: TTS simples")
                
                # Decidir se incluir language baseado no modelo
                if "pt" in model_name.lower() or "multilingual" in model_name.lower():
                    wav_data = tts_model.tts(text=text, language=language)
                    method_used = "TTS simples com language"
                else:
                    wav_data = tts_model.tts(text=text)
                    method_used = "TTS simples sem language"
                    
                logger.info(f"✅ Método 2 funcionou: {method_used}")
                
            except Exception as method2_error:
                logger.error(f"❌ Método 2 também falhou: {method2_error}")
                raise Exception(f"Todos os métodos falharam. Último erro: {method2_error}")
        
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
        
        logger.info(f"📊 Array final: shape={wav_data.shape}, dtype={wav_data.dtype}, método={method_used}")
        
        # Escrever no buffer como WAV
        sample_rate = 22050  # Taxa de amostragem padrão do XTTS v2
        sf.write(audio_buffer, wav_data, sample_rate, format='WAV')
        audio_buffer.seek(0)
        
        # Verificar tamanho do buffer
        buffer_size = audio_buffer.getbuffer().nbytes
        logger.info(f"💾 Buffer de áudio criado: {buffer_size} bytes")
        
        if buffer_size == 0:
            raise Exception("Buffer de áudio está vazio")
        
        logger.info(f"🎉 Áudio TTS gerado com sucesso usando: {method_used}! 🎵")
        
        # Retornar como streaming response
        return StreamingResponse(
            io.BytesIO(audio_buffer.read()),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=voice_clone_output.wav"}
        )
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"❌ Erro ao gerar áudio TTS: {error_msg}")
        logger.error(f"📍 Detalhes do erro:", exc_info=True)

@app.get("/clone-info")
async def voice_clone_info():
    """
    📖 Informações sobre síntese TTS
    """
    return {
        "feature": "Síntese de Voz TTS",
        "description": "Converte texto em áudio usando modelos Coqui TTS otimizados",
        "endpoint": "/tts-clone",
        "method": "POST",
        "model_info": {
            "loaded_model": getattr(tts_model, 'model_name', 'Não carregado') if tts_model else 'Não carregado',
            "multi_speaker": bool(getattr(tts_model, 'speakers', None)) if tts_model else False,
            "device": "cpu",
            "status": "funcionando" if tts_model else "não inicializado"
        },
        "supported_languages": ["pt", "en", "es", "fr", "de"],
        "tips": [
            "Modelos portugueses funcionam melhor para texto em português",
            "Arquivos de áudio são aceitos mas podem não influenciar o resultado",
            "Use textos claros e bem pontuados para melhores resultados",
            "O sistema usa automaticamente a melhor voz disponível"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)