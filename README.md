# llama.cpp CUDA builds

## clone llama.cpp



### building docker

```
docker build -t llama-cuda-builder .
```

### running

```
docker run --rm -v $(pwd):/src -a stdout -a stderr llama-cuda-builder
```


The build artifacts are in `llama.cpp/build/bin`
