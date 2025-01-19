FROM python:3.11.2-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/0.5.14/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app
COPY ["pyproject.toml", "./"]

RUN uv pip install --system -r "pyproject.toml"

COPY ["predict.py", "model_C=1.0.bin", "./"]

EXPOSE 9696

CMD ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]