# Multi-stage Dockerfile for Medical Aesthetics Extraction Engine
# Optimized for both development and production environments

FROM denoland/deno:1.40.0 as base

# Set working directory
WORKDIR /app

# Copy dependency files
COPY deno.json package.json ./

# Install dependencies
RUN deno cache deno.json

# Development stage
FROM base as development

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run in development mode
CMD ["deno", "run", "--allow-all", "--watch", "src/index.ts"]

# Production stage
FROM base as production

# Copy source code
COPY . .

# Create non-root user
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 deno

# Change ownership
RUN chown -R deno:nodejs /app
USER deno

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run in production mode
CMD ["deno", "run", "--allow-all", "src/index.ts"]