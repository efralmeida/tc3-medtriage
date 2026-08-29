# tc3-medtriage

API de triagem automatica de laudos medicos para classificar urgência em 3 classes:
- normal
- atenção
- urgente

Este repositório atende a Etapa 1 do Tech Challenge (FIAP ML Engineering):
- decisao arquitetural documentada
- API FastAPI funcional para inferencia
- empacotamento em Docker
- medicao de latencia baseline

## 1) Decisão Arquitetural (Nuvem + Batch vs Real-Time)

### Provedor escolhido
AWS.

### Estrategia de deploy
- API de inferencia em tempo real com FastAPI (deploy em ECS Fargate ou EKS).
- Artefato do modelo armazenado em objeto versionado (S3).
- Pipeline de treino separado da inferencia (Airflow).

### Justificativa Batch vs Real-Time
Escolha principal: real-time.

Motivos de negócio:
- triagem hospitalar exige resposta imediata por paciente
- suporte a decisão clínica no momento da entrada
- reduz fila manual de priorização

Motivos técnicos:
- modelo linear com TF-IDF tem inferencia rapida
- API stateless simplifica escalabilidade horizontal

Onde batch entra no desenho:
- retreino agendado
- geração de relatórios gerenciais
- reprocessamento histórico

## 2) API FastAPI, empacotamento Docker, medição de latência

- src/medtriage/main.py: entrada da API
- src/medtriage/app_factory.py: Factory da aplicacao (design pattern)
- src/medtriage/model_service.py: camada de inferência e confiança
- src/medtriage/schemas.py: contratos da API
- scripts/benchmark_api_latency.py: benchmark baseline de latência
- Dockerfile: empacotamento da API

## 3) CI/CD e treinamento orquestrado

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

O Airflow pode ser iniciado junto com o stack Docker. Ele usa PostgreSQL para os
metadados, um webserver para a interface e um scheduler para executar as DAGs:

```powershell
docker compose up --build airflow-init airflow-webserver airflow-scheduler postgres
```

Acesse `http://127.0.0.1:8080` e entre com `admin` / `admin`. A DAG
`medtriage_training` aparecera na interface. Ative-a e use **Trigger DAG** para
executar manualmente. Os diretorios `dags/`, `data/` e `models/` sao montados no
container, permitindo que as tasks leiam o dataset e salvem o modelo no projeto.

Para testar a DAG sem abrir a interface:

```powershell
docker compose run --rm airflow-scheduler airflow dags test medtriage_training 2026-01-01
```

O comando abaixo encerra os servicos, preservando o banco de metadados no volume:

```powershell
docker compose down
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

Resposta ilustrativa (a classe, a confianca e a latencia variam conforme o
modelo carregado e o ambiente de execucao):

```json
{
	"urgency": "<classe prevista>",
	"confidence": 0.0,
	"latency_ms": 0.0
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

## 6) Medição de Latência Baseline

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
- `avg_latency_ms` representa a latencia média baseline.
- `p95_latency_ms` captura cauda de latência e e um indicador mais robusto para SLO inicial.

## 7) Monitoramento e Observabilidade

A API é instrumentada com `prometheus_client`:

- `medtriage_http_requests_total`: contador de requisições por metodo, endpoint e status HTTP.
- `medtriage_http_request_latency_seconds`: histograma de latência por método e endpoint.
- `GET /metrics`: expõe as métricas no formato Prometheus.

O comando abaixo constroi as imagens da API e do Airflow e sobe todo o stack
local: API, Prometheus, Grafana, PostgreSQL e servicos Airflow.

```bash
docker compose up --build
```

Servicos expostos:
- API: http://127.0.0.1:8000
- Prometheus: http://127.0.0.1:9090
- Grafana: http://127.0.0.1:3000 (usuario `admin`, senha `admin`)
- Airflow: http://127.0.0.1:8080 (usuario `admin`, senha `admin`)

O dashboard `MedTriage - API Overview` é provisionado automaticamente em
`monitoring/grafana/dashboards/medtriage_overview.json` e contém:
- total de requisições
- latência média e p95
- taxa de erro (%)
- requisições por endpoint

Para gerar trafego de teste, use o benchmark da seção 6 apontando para
`http://127.0.0.1:8000`.

Para derrubar todo o stack:

```bash
docker compose down
```

## 8) Otimização com ONNX Runtime

O pipeline `TF-IDF + LogisticRegression` é exportado para ONNX e executado com
ONNX Runtime em CPU. O comando abaixo mede `avg`, `p50` e `p95` apos warm-up e
grava o comparativo em `models/model_latency_comparison.csv`. Se o CSV informado
tiver a coluna `urgency`, o comando tambem registra a acuracia de cada backend:

O exportador preserva a normalização de acentos do modelo Joblib antes de enviá-la
ao ONNX, pois `skl2onnx` nao converte `strip_accents="unicode"` diretamente.

O arquivo `data/raw/test.dat` é um conjunto de inferência sem rótulos: ele
comeca diretamente com o texto, ao contrario de `train.dat`, que possui o
codigo numerico da classe no inicio de cada linha. Por isso, a acuracia no
comparativo e `n/a` para esse arquivo. O benchmark registra
`prediction_agreement`, que mede a concordância entre Joblib e ONNX e valida a
equivalência das predições sem inventar rótulos.

Para obter uma acurácia com ground truth, gere uma validação estratificada a
partir do conjunto rotulado de treino:

```powershell
$env:PYTHONPATH = "src"
poetry run python scripts/create_validation_split.py
```

O comando gera `data/processed/train_split.csv` e
`data/processed/validation.csv` usando `random_state=42`. O modelo usado na
avaliacao deve ser treinado com `train_split.csv`, e o benchmark deve receber
`--test-data data/processed/validation.csv`. O `data/processed/test.csv`
continua reservado para inferencia sem rotulos.

PowerShell:

```powershell
$env:PYTHONPATH = "src"
poetry run python scripts/optimize_model.py --iterations 200
```

Linux/macOS:

```bash
PYTHONPATH=src poetry run python scripts/optimize_model.py --iterations 200
```

Arquivos gerados:
- `models/text_classifier_best.onnx`: artefato otimizado.
- `models/model_latency_comparison.csv`: acuracia, concordancia entre backends e
	latencias do Joblib e ONNX Runtime.

Para executar a API com o modelo otimizado fora do Compose:

```bash
MODEL_PATH=models/text_classifier_best.onnx PYTHONPATH=src \
poetry run uvicorn medtriage.main:app --host 0.0.0.0 --port 8000
```

O Compose ja aponta para `text_classifier_best.onnx`; gere o artefato antes de
executar `docker compose up --build`. O fallback Joblib continua disponivel
quando `MODEL_PATH` aponta para `text_classifier_best.joblib`.

O servico da API possui um healthcheck no Compose que executa uma requisicao
real em `/predict`. Assim, Prometheus so inicia depois que o modelo ONNX foi
carregado e respondeu com sucesso a uma inferencia.

## 9) Entregáveis

- API em Docker funcional
- decisao arquitetural registrada neste README
- procedimento de benchmark baseline de latencia documentado
- workflow CI com lint e testes automatizados
- DAG Airflow com ingestao, treinamento e salvamento do modelo
- instrumentacao Prometheus na API e stack Docker Compose (API + Prometheus + Grafana)
- dashboard Grafana com >= 3 paineis provisionado automaticamente
- modelo ONNX Runtime, validacao de equivalencia e comparativo de latencia

