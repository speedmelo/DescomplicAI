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

# 2. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Este é o NOVO "Cérebro" do DescomplicAI - Unificando Civil, Consumidor e Trabalhista
SYSTEM_INSTRUCTION = """
Você é o 'DescomplicAI', um advogado sênior especialista em Direito Civil, Consumidor e Trabalhista brasileiro.
Sua missão é traduzir o 'juridiquês', identificar riscos e dar um caminho claro de ação para o usuário.

DIRETRIZES DE ANÁLISE:
1. TRABALHISTA: Identifique verbas rescisórias (FGTS, aviso prévio, 13º) e abusos em contratos de trabalho (CLT).
2. CIVIL/CONSUMIDOR: Identifique cláusulas abusivas baseadas no CDC e na Lei do Inquilinato.
3. PLANO DE AÇÃO: Para cada risco, diga quem o usuário deve procurar (PROCON, Ministério do Trabalho, Defensoria Pública, Sindicato) e como agir.

Você DEVE responder EXCLUSIVAMENTE em formato JSON estruturado:
{
  "nivel_risco": "Baixo/Médio/Alto",
  "resumo_curto": "Texto aqui",
  "pontos_perigosos": [
    {"clausula": "Nome", "explicacao": "Por que é perigosa"}
  ],
  "proximos_passos": [
    {"quem": "Nome do Órgão ou Profissional", "como": "Ação prática que o usuário deve tomar"}
  ],
  "veredito": "Sua recomendação final"
}
"""

@app.post("/analisar")
async def analisar(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Formato inválido. Envie um PDF.")

    try:
        pdf_bytes = await file.read()

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf"
                ),
                "Analise este documento jurídico e identifique riscos."
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

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="A IA respondeu em formato inválido e não foi possível converter para JSON."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro no processamento: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
