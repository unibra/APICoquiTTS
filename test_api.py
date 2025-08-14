"""
Script de teste para verificar o funcionamento da API TTS
"""

import requests
import json
import time

# Configuração
API_BASE_URL = "http://localhost:8888"  # ou "http://localhost" se usando nginx

def test_health():
    """Testar endpoint de health check"""
    print("🔍 Testando health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Erro no health check: {e}")
        return False

def test_models():
    """Testar listagem de modelos"""
    print("\n📋 Testando listagem de modelos...")
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            models = response.json()["available_models"]
            print(f"✅ {len(models)} modelos disponíveis")
            return True
        else:
            print(f"❌ Erro: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erro ao listar modelos: {e}")
        return False

def test_tts_simple():
    """Testar TTS com texto simples"""
    print("\n🎵 Testando TTS simples...")
    try:
        data = {
            "text": "Hello world! This is a test of the Coqui TTS service."
        }
        
        print(f"Enviando texto: {data['text']}")
        start_time = time.time()
        
        response = requests.post(f"{API_BASE_URL}/tts", json=data)
        
        if response.status_code == 200:
            duration = time.time() - start_time
            print(f"✅ Áudio gerado em {duration:.2f} segundos")
            
            # Salvar arquivo de áudio
            with open("test_output.wav", "wb") as f:
                f.write(response.content)
            print("💾 Áudio salvo como 'test_output.wav'")
            return True
        else:
            print(f"❌ Erro: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste TTS: {e}")
        return False

def test_tts_custom():
    """Testar TTS com modelo personalizado"""
    print("\n🎵 Testando TTS com modelo personalizado...")
    try:
        data = {
            "text": "This is a test with a custom model configuration.",
            "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
            "language": "en"
        }
        
        print(f"Modelo: {data['model_name']}")
        print(f"Texto: {data['text']}")
        
        response = requests.post(f"{API_BASE_URL}/tts", json=data)
        
        if response.status_code == 200:
            print("✅ TTS personalizado funcionando!")
            
            # Salvar arquivo
            with open("test_custom.wav", "wb") as f:
                f.write(response.content)
            print("💾 Áudio salvo como 'test_custom.wav'")
            return True
        else:
            print(f"❌ Erro: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste personalizado: {e}")
        return False

def main():
    """Executar todos os testes"""
    print("🚀 Iniciando testes da API TTS...")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("Listagem de Modelos", test_models),
        ("TTS Simples", test_tts_simple),
        ("TTS Personalizado", test_tts_custom)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n▶️  {test_name}")
        result = test_func()
        results.append((test_name, result))
    
    # Resumo dos resultados
    print("\n" + "=" * 50)
    print("📊 RESUMO DOS TESTES:")
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"  {test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\n🎯 Resultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 Todos os testes passaram! API está funcionando corretamente.")
    else:
        print("⚠️  Alguns testes falharam. Verifique os logs acima.")

if __name__ == "__main__":
    main()