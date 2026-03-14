import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

# 1. Carregar ambiente
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("ERRO: Variável GOOGLE_API_KEY não encontrada no arquivo .env")

client = genai.Client(api_key=GOOGLE_API_KEY)

app = FastAPI(title="DescomplicAI API")

# 2. CORS - ESSENCIAL para o site falar com o Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrução do Sistema (Mantida sua lógica excelente!)
SYSTEM_INSTRUCTION = """
Você é o 'DescomplicAI', um advogado sênior especialista em Direito Civil, Consumidor e Trabalhista brasileiro.
Sua missão é traduzir o 'juridiquês', identificar riscos e dar um caminho claro de ação para o usuário.

DIRETRIZES DE ANÁLISE:
1. TRABALHISTA: Identifique verbas rescisórias e abusos em contratos.
2. CIVIL/CONSUMIDOR: Identifique cláusulas abusivas (CDC e Lei do Inquilinato).
3. PLANO DE AÇÃO: Diga quem o usuário deve procurar e como agir.

Responda EXCLUSIVAMENTE em JSON:
{
  "nivel_risco": "Baixo/Médio/Alto",
  "resumo_curto": "Texto",
  "pontos_perigosos": [{"clausula": "Nome", "explicacao": "Risco"}],
  "proximos_passos": [{"quem": "Órgão", "como": "Ação"}],
  "veredito": "Recomendação"
}
"""

# Rota Raiz (Evita erro 404 no Render)
@app.get("/")
async def root():
    return {"status": "online", "message": "API Descomplic.AI ativa!"}

# ROTA DE ANÁLISE - MUDAMOS PARA /analyze (PADRÃO)
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Formato inválido. Envie um PDF.")

    try:
        pdf_bytes = await file.read()

        # CORREÇÃO DO MODELO: Usando gemini-2.0-flash (O mais rápido e estável)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf"
                ),
                "Analise este documento jurídico e identifique riscos conforme suas instruções."
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json"
            )
        )

        analise_json = json.loads(response.text)

        return {
            "status": "sucesso",
            "dados": analise_json
        }

    except Exception as e:
        print(f"ERRO: {str(e)}") # Log para você ver no painel do Render
        raise HTTPException(
            status_code=500,
            detail=f"Erro no processamento: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    # O Render usa a variável de ambiente PORT, se não achar, usa 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
