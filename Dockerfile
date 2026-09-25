FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY configs ./configs

# Mount the dataset at /app/data/raw and collect outputs from /app/results:
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" eeg-unlearning
ENTRYPOINT ["eeg-unlearn"]
CMD ["run", "--config", "configs/default.yaml"]
