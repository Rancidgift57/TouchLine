# TouchLine backend — FastAPI + WebSocket match streamer.
# Build & run:  docker build -t touchline-api . && docker run -p 8080:8080 touchline-api

FROM python:3.12-slim

WORKDIR /app

# System deps for torch/scipy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY engine/ engine/
COPY api/ api/
COPY db/ db/
COPY ml/ ml/

EXPOSE 8080

CMD ["uvicorn", "api.match_stream:app", "--host", "0.0.0.0", "--port", "8080"]
