from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import logging
from typing import Optional
import uvicorn

try:
    from TTS.api import TTS
except ImportError:
    TTS = None

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Coqui TTS API",
    description="Serviço de Text-to-Speech usando Coqui TTS",
    version="1.0.0"
)

# Modelo de request
class TTSRequest(BaseModel):
    text: str
    model_name: Optional[str] = "tts_models/en/ljspeech/tacotron2-DDC"
    speaker: Optional[str] = None
    language: Optional[str] = "en"

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
        tts_model = TTS(model_name=default_model)
        logger.info("Modelo TTS carregado com sucesso!")
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
    return {
        "status": "healthy",
        "tts_available": tts_model is not None
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
            current_tts = TTS(model_name=request.model_name)
        
        if current_tts is None:
            raise HTTPException(status_code=500, detail="Modelo TTS não está carregado")
        
        # Gerar áudio
        logger.info(f"Gerando áudio para texto: {request.text[:50]}...")
        
        # Criar buffer em memória para o áudio
        audio_buffer = io.BytesIO()
        
        # Parâmetros para TTS
        tts_kwargs = {}
        if request.speaker:
            tts_kwargs["speaker"] = request.speaker
        if request.language:
            tts_kwargs["language"] = request.language
        
        # Gerar áudio e salvar no buffer
        wav_data = current_tts.tts(text=request.text, **tts_kwargs)
        
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