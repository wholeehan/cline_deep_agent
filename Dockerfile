FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY skills/ skills/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Create workspace and log directories
RUN mkdir -p workspace logs

CMD ["python", "-m", "src.cli"]
