# to build this docker image:
# docker build -t llama-cuda-builder .
# to run:    
# docker run --rm -e -v $(pwd):/src -v $(pwd)/build:/build -a stdout -a stderr llama-cuda-builder
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 AS llama-gpu-cuda-12-builder
LABEL maintainer="hybridgroup"

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake libcurl4 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY ./builder.sh /src/

CMD ["/src/builder.sh"]
