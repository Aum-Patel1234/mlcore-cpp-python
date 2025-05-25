FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install .

CMD ["python3"]

