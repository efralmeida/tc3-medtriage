# tc3-medtriage

API de triagem automatica de laudos medicos para classificar urgencia em 3 classes:
- normal
- atencao
- urgente

Este repositorio atende a Etapa 1 do Tech Challenge (FIAP ML Engineering):
- decisao arquitetural documentada
- API FastAPI funcional para inferencia
- empacotamento em Docker
- medicao de latencia baseline

## 1) Decisao Arquitetural (Nuvem + Batch vs Real-Time)

### Provedor escolhido
AWS.

### Estrategia de deploy
- API de inferencia em tempo real com FastAPI (deploy em ECS Fargate ou EKS).
- Artefato do modelo armazenado em objeto versionado (S3).
- Pipeline de treino separado da inferencia (Airflow na etapa 2).

### Justificativa Batch vs Real-Time
Escolha principal: real-time.

Motivos de negocio:
- triagem hospitalar exige resposta imediata por paciente
- suporte a decisao clinica no momento da entrada
- reduz fila manual de priorizacao

Motivos tecnicos:
- modelo linear com TF-IDF tem inferencia rapida
- API stateless simplifica escalabilidade horizontal

Onde batch entra no desenho:
- retreino agendado
- geracao de relatorios gerenciais
- reprocessamento historico

## 2) Estrutura da Etapa 1

- src/medtriage/main.py: entrada da API
- src/medtriage/app_factory.py: Factory da aplicacao (design pattern)
- src/medtriage/model_service.py: camada de inferencia e confianca
- src/medtriage/schemas.py: contratos da API
- scripts/benchmark_api_latency.py: benchmark baseline de latencia
- Dockerfile: empacotamento da API

## 3) Como Executar Localmente

Pre-requisitos:
- Python 3.12+
- Poetry instalado
- modelo salvo em models/text_classifier_best.joblib

Instalacao:

```bash
poetry install
```

Subir API:

```bash
poetry run uvicorn medtriage.main:app --app-dir src --host 0.0.0.0 --port 8000
```

### Teste rapido

Health:

```bash
curl http://127.0.0.1:8000/health
```

Predicao:

```bash
curl -X POST http://127.0.0.1:8000/predict \
	-H "Content-Type: application/json" \
	-d "{\"text\":\"Paciente com dor toracica intensa e dispneia subita\"}"
```

Resposta esperada (exemplo):

```json
{
	"urgency": "urgente",
	"confidence": 0.74,
	"latency_ms": 7.91
}
```

## 4) Docker

Build da imagem:

```bash
docker build -t medtriage-api:baseline .
```

Run do container:

```bash
docker run --rm -p 8000:8000 medtriage-api:baseline
```

Se quiser apontar para outro artefato de modelo:

```bash
docker run --rm -p 8000:8000 \
	-e MODEL_PATH=/app/models/text_classifier_best.joblib \
	medtriage-api:baseline
```

## 5) Medicao de Latencia Baseline

Com a API no ar (local ou Docker), execute:

```bash
poetry run python scripts/benchmark_api_latency.py --url http://127.0.0.1:8000 --requests 50
```

Saida esperada:

```text
requests=50
errors=0
avg_latency_ms=...
p50_latency_ms=...
p95_latency_ms=...
```

Interpretacao:
- `avg_latency_ms` representa a latencia media baseline.
- `p95_latency_ms` captura cauda de latencia e e um indicador mais robusto para SLO inicial.

## 6) Entregavel da Etapa 1

- API em Docker funcional
- decisao arquitetural registrada neste README
- procedimento de benchmark baseline de latencia documentado
