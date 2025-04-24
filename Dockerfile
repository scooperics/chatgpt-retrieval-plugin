# =============================================================================
# Requirements stage – exports requirements.txt with Poetry
# =============================================================================
FROM python:3.10 AS requirements-stage

WORKDIR /tmp

# Upgrade pip
RUN pip install --upgrade pip

# Install Poetry **plus the plugin that provides `poetry export`**
RUN pip install poetry poetry-plugin-export   # ← minimal fix

# Copy project metadata
COPY ./pyproject.toml ./poetry.lock* /tmp/

# Export requirements.txt (now works because plugin is installed)
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# =============================================================================
# Runtime stage – installs deps and starts the app
# =============================================================================
FROM python:3.10

# OS-level build tools (needed for some wheels)
RUN apt-get update && apt-get install -y build-essential

# Install Rust toolchain (many crypto/pyarrow wheels need it)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Upgrade pip
RUN pip install --upgrade pip

WORKDIR /code

# Install Python dependencies
COPY --from=requirements-stage /tmp/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy application code
COPY . /code/

# Start the FastAPI app (PORT is set by most PaaS providers)
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-${WEBSITES_PORT:-8080}}"]
