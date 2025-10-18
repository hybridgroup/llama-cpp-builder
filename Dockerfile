# to build this docker image:
# docker build -t llama-cuda-builder .
# to run:    
# docker run --rm -v $(pwd):/src -a stdout -a stderr llama-cuda-builder
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS llama-gpu-cuda-12-builder
LABEL maintainer="hybridgroup"

RUN apt-get update && \
    apt-get install -y build-essential cmake git libcurl4-openssl-dev libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY ./builder.sh /src/builder.sh

CMD ["/src/builder.sh"]
