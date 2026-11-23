FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends procps coreutils \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app/pipeline
CMD ["python", "status.py"]
