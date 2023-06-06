FROM python:3.9.16-slim-bullseye


ENV PYTHONUNBUFFERED True

RUN apt-get update && \
	apt-get install --no-install-recommends -y ffmpeg git && \
	pip install --no-cache-dir --upgrade pip setuptools wheel && \
	mkdir -p /opt/disco-whisper

WORKDIR /opt/disco-whisper

COPY ./src ./pyproject.toml /opt/disco-whisper/

ARG MODEL=base.en

RUN pip install --no-cache-dir -e . && \
	python -c "from whisper_jax import FlaxWhisper;import jax.numpy as jnp;FlaxWhisperPipline('openai/whisper-${MODEL}', dtype=jnp.bfloat16)"

CMD ["bot"]
