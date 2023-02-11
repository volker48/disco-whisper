FROM python:3.10.10-slim-bullseye

ENV PYTHONUNBUFFERED True

RUN apt-get update && apt-get install -y ffmpeg

RUN pip install -U pip setuptools

RUN mkdir -p /opt/disco-whisper

WORKDIR /opt/disco-whisper

COPY ./src ./pyproject.toml ./setup.py /opt/disco-whisper/

RUN pip install -e .

CMD ["bot"]