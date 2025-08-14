#!/bin/bash

# Script para testar a API localmente
echo "🧪 Testando API TTS..."

# Verificar se a API está funcionando
echo "1. Testando health check..."
curl -f http://localhost:8888/health || {
    echo "❌ API não está respondendo"
    exit 1
}

echo "✅ Health check OK!"

# Testar endpoint TTS
echo "2. Testando conversão TTS..."
curl -X POST "http://localhost:8888/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of the TTS service!"
  }' \
  --output test_audio.wav

if [ -f test_audio.wav ]; then
    echo "✅ Áudio gerado com sucesso: test_audio.wav"
    ls -lh test_audio.wav
else
    echo "❌ Falha ao gerar áudio"
    exit 1
fi

echo "🎉 Todos os testes passaram!"