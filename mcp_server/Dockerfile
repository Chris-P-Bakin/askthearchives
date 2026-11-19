FROM python:3.12-slim

RUN useradd -m archives
WORKDIR /app

COPY mcp_server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model so the container needs no network at runtime
ENV FASTEMBED_CACHE_PATH=/opt/fastembed
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')" \
    && chmod -R a+rX /opt/fastembed

COPY mcp_server/ /app/mcp_server/

USER archives
ENV ARCHIVES_DB=/index/corpus.db ARCHIVES_DATA=/data ARCHIVES_HOST=0.0.0.0 ARCHIVES_PORT=8800
EXPOSE 8800
CMD ["python", "-m", "mcp_server.server"]
