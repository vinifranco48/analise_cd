# Dockerfile corrigido: instala dependências a partir de pyproject.toml + uv.lock
FROM python:3.11-slim

# Não gerar .pyc e não bufferizar stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# (Opcional mas útil) instalar ferramentas para compilar dependências nativas
# Remova estas linhas se tem certeza que não precisa compilar binários nativos.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia só os metadados primeiro para aproveitar cache de camadas
COPY pyproject.toml uv.lock ./

# Atualiza pip e instala uv
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir uv

# Força o uv a instalar no Python do sistema (prefix /usr/local) em vez de criar .venv.
# UV_BREAK_SYSTEM_PACKAGES=true permite sobrescrever pacotes do sistema (útil em CI/container).
ENV UV_PROJECT_ENVIRONMENT=/usr/local
ENV UV_BREAK_SYSTEM_PACKAGES=true

# Sincroniza dependências a partir do lockfile (uv.lock)
# Usamos `--locked` para garantir que o lockfile seja respeitado (falha se estiver desatualizado).
RUN uv sync --locked

# Copia o restante do projeto
COPY . .

# (Opcional) cria um usuário não-root e dá posse do diretório /app
RUN useradd -m appuser \
 && chown -R appuser:appuser /app

USER appuser

# Comando padrão
CMD ["python", "main.py"]
