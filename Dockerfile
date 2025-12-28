# Multi-stage build for production ML application
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements-prod.txt

# Production stage
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Add local bin and system paths to PATH
# /root/.local/bin: for pip-installed command-line tools
# /usr/local/bin: for system-installed binaries
ENV PATH=/root/.local/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin

# Set Python path for module discovery
# /app: root application directory
# /app/src: source code directory with main modules
ENV PYTHONPATH=/app:/app/src

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command (can be overridden)
CMD ["python", "src/entrypoint/inference.py"]