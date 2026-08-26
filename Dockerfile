FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY src ./src
COPY models ./models

RUN pip install --upgrade pip
RUN pip install \
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

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "medtriage.main:app", "--host", "0.0.0.0", "--port", "8000"]
