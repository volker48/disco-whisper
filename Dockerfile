FROM python:3.10.10-slim-bullseye

ENV PYTHONUNBUFFERED True

RUN apt-get update && \
	apt-get install --no-install-recommends -y ffmpeg && \
	pip install --no-cache-dir -U pip setuptools && \
	mkdir -p /opt/disco-whisper

WORKDIR /opt/disco-whisper

COPY ./src ./pyproject.toml ./setup.py /opt/disco-whisper/

RUN pip install --no-cache-dir -e .

CMD ["bot"]
