FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System libraries required by numerical and image-processing packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies before copying source code to reuse the build cache.
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Both backend and frontend use this source code.
COPY src ./src
COPY app.py .

EXPOSE 8000 8501

# Default command: start the backend.
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]