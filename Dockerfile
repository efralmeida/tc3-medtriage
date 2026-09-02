FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN python -m venv /opt/venv \
	&& /opt/venv/bin/pip install --upgrade pip \
	&& /opt/venv/bin/pip install \
	fastapi \
	"uvicorn[standard]" \
	scikit-learn \
	pandas \
	matplotlib \
	joblib \
	prometheus-client \
	numpy \
	onnx \
	onnxruntime \
	skl2onnx

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY src ./src
COPY models ./models

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "medtriage.main:app", "--host", "0.0.0.0", "--port", "8000"]
