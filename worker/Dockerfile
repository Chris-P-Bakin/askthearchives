# Unit worker image: llama.cpp CUDA server + the python worker.
# Build from the repo root:  docker build -f worker/Dockerfile -t naca-worker .
# Model GGUF + mmproj are volume-mounted at /models (NOT baked: 9.4 GB).
FROM ghcr.io/ggml-org/llama.cpp:server-cuda
RUN apt-get update && apt-get install -y --no-install-recommends python3 curl boinc-client \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY worker/unit_worker.py worker/entry.sh /worker/
COPY pipeline/guard.py /worker/guard.py
COPY pipeline/prompts/transcribe_v2.txt /worker/transcribe_v2.txt
COPY worker/boinc/ /worker/boinc/
RUN chmod +x /worker/entry.sh /worker/boinc/naca_vision
ENV NACA_VLM_URL=http://127.0.0.1:8080
ENTRYPOINT ["/worker/entry.sh"]
CMD ["standalone"]
