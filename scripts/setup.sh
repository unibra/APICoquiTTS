#!/bin/bash

# Script de configuração inicial
echo "🚀 Configurando serviço Coqui TTS..."

# Criar diretórios necessários
mkdir -p models output

# Verificar se Docker está instalado
if ! command -v docker &> /dev/null; then
    echo "❌ Docker não está instalado. Por favor, instale Docker primeiro."
    exit 1
fi

# Verificar se Docker Compose está instalado
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose não está instalado. Por favor, instale Docker Compose primeiro."
    exit 1
fi

# Verificar se runtime NVIDIA está configurado
if ! docker info 2>/dev/null | grep -q nvidia; then
    echo "⚠️  Runtime NVIDIA pode não estar configurado. Verifique se nvidia-docker2 está instalado."
fi

echo "✅ Docker e Docker Compose encontrados!"

# Construir e iniciar serviços
echo "🔨 Construindo imagem Docker..."
docker-compose build

echo "🚀 Iniciando serviços..."
docker-compose up -d

echo "⏳ Aguardando serviços ficarem prontos..."
sleep 10

# Verificar se os serviços estão funcionando
echo "🔍 Verificando saúde dos serviços..."
if curl -f http://localhost:8888/health > /dev/null 2>&1; then
    echo "✅ API TTS está funcionando!"
    echo "📖 Documentação disponível em: http://localhost:8888/docs"
    echo "🌐 Serviço disponível em: http://localhost:8888"
else
    echo "❌ Serviço não está respondendo. Verificando logs..."
    docker-compose logs tts-api
fi

echo "🎉 Configuração concluída!"
echo ""
echo "Para testar a API, execute:"
echo "  python test_api.py"
echo ""
echo "Para parar os serviços:"
echo "  docker-compose down"