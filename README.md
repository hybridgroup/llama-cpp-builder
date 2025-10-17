# llama.cpp CUDA builds

## clone llama.cpp



### docker

```
docker build -t llama-cuda-builder .
```

### building

```
docker run --rm -v $(pwd):/src -a stdout -a stderr llama-cuda-builder
```
