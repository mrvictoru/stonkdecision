FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /code

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    python3.10 \
    python3-pip \
    python3-dev \
    libglib2.0-0

COPY ./requirements.txt ./

COPY ./torch_req.txt ./

RUN pip install --no-cache-dir -r torch_req.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src

CMD ["python", "./src/main.py"]