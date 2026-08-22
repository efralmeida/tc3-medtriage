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

## 3) Etapa 2: CI/CD e treinamento orquestrado

O workflow `.github/workflows/ci.yml` executa Ruff e Pytest em todo push e pull
request. A DAG `dags/medtriage_training.py` possui duas tarefas:

1. `ingest_data`: valida e copia `data/processed/train.csv` para a area de trabalho do Airflow.
2. `train_and_save`: treina o pipeline TF-IDF + Regressao Logistica e salva o artefato em `models/text_classifier_best.joblib`.

Para validar a qualidade localmente:

```bash
poetry install --with dev
poetry run ruff check src tests dags
poetry run pytest
```

Em um ambiente Airflow, configure o repositorio como volume de DAGs e execute:

```bash
airflow dags test medtriage_training 2026-01-01
```

## 4) Como Executar Localmente

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

## 5) Docker

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

## 6) Medicao de Latencia Baseline

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

## 7) Entregaveis

- API em Docker funcional
- decisao arquitetural registrada neste README
- procedimento de benchmark baseline de latencia documentado
- workflow CI com lint e testes automatizados
- DAG Airflow com ingestao, treinamento e salvamento do modelo
