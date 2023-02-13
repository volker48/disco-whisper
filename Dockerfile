FROM python:3.10.10-slim-bullseye


ENV PYTHONUNBUFFERED True

RUN apt-get update && \
	apt-get install --no-install-recommends -y ffmpeg && \
	pip install --no-cache-dir -U pip setuptools && \
	mkdir -p /opt/disco-whisper

WORKDIR /opt/disco-whisper

COPY ./src ./pyproject.toml ./setup.py /opt/disco-whisper/

ARG MODEL=base.en

RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu --no-cache-dir -e . && \
	python -c "import whisper;whisper.load_model('${MODEL}')"

CMD ["bot"]
