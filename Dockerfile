# Multi-stage Dockerfile for Medical Aesthetics Extraction Engine
# Elite Technical Consortium Implementation

# Stage 1: Base image with Deno runtime
FROM denoland/deno:1.40.0 AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libxss1 \
    libxtst6 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Development environment
FROM base AS development

# Copy package files
COPY deno.json package.json ./

# Cache dependencies
RUN deno cache deno.json

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY schemas/ ./schemas/

# Set permissions
RUN chmod +x src/index.ts

# Expose ports
EXPOSE 8080 9090 9229

# Development command
CMD ["deno", "run", "--allow-all", "--watch", "src/index.ts"]

# Stage 3: Build stage
FROM base AS build

# Copy package files
COPY deno.json package.json ./

# Cache dependencies
RUN deno cache deno.json

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY schemas/ ./schemas/

# Build the application
RUN deno compile --allow-all --output medical-extraction src/index.ts

# Stage 4: Production runtime
FROM debian:bookworm-slim AS production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r medical && useradd -r -g medical medical

# Set working directory
WORKDIR /app

# Copy built application
COPY --from=build /app/medical-extraction /app/medical-extraction
COPY --from=build /app/config /app/config
COPY --from=build /app/schemas /app/schemas

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/ml-models && \
    chown -R medical:medical /app

# Switch to non-root user
USER medical

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 9090

# Production command
CMD ["./medical-extraction"]

# Stage 5: ML Models stage
FROM base AS ml-models

# Install Python for ML model processing
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install ML dependencies
RUN pip3 install --no-cache-dir \
    torch \
    transformers \
    sentence-transformers \
    scikit-learn \
    numpy \
    pandas

# Download and prepare ML models
WORKDIR /app/ml-models

# Download Persian medical NLP models
RUN python3 -c "
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# Download Persian medical models
print('Downloading Persian medical NLP models...')
tokenizer = AutoTokenizer.from_pretrained('HooshvareLab/bert-fa-base-uncased')
model = AutoModel.from_pretrained('HooshvareLab/bert-fa-base-uncased')

# Download English medical models
print('Downloading English medical NLP models...')
en_model = SentenceTransformer('all-MiniLM-L6-v2')

# Download medical entity recognition models
print('Downloading medical NER models...')
medical_ner = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

print('Models downloaded successfully!')
"

# Stage 6: Final production image with ML models
FROM production AS production-with-ml

# Copy ML models
COPY --from=ml-models /app/ml-models /app/ml-models

# Update permissions
RUN chown -R medical:medical /app/ml-models

# Environment variables for ML models
ENV ML_MODELS_PATH=/app/ml-models
ENV PERSIAN_MODEL_PATH=/app/ml-models/persian-medical
ENV ENGLISH_MODEL_PATH=/app/ml-models/english-medical
ENV NER_MODEL_PATH=/app/ml-models/medical-ner

# Labels for metadata
LABEL maintainer="Elite Technical Consortium"
LABEL version="1.0.0"
LABEL description="Medical Aesthetics Extraction Engine with Advanced NLP"
LABEL org.opencontainers.image.source="https://github.com/elite-consortium/medical-extraction"
LABEL org.opencontainers.image.licenses="MIT"

# Multi-architecture support
# This Dockerfile supports both AMD64 and ARM64 architectures
