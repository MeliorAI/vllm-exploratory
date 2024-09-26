# LLM-exploratory


# HowTo

## Install

```shell
pdm install
```


## Run

### Batch

Generate several generations reading from an input file (one prompt per line)

```shell
python -m llm.xplore batch run -m <modelID> -i <input-file>
```

### Serve

Starts a vLLM server exposing an OpenAI-like API

```shell
python -m llm.xplore serve

# Somewhat equivalent to:
# (To see all options check 'vllm serve --help'):
vllm serve <modelID> \
    --chat-template templates/template_chatml.jinja \
    --device auto \
    --max-model-len 400 \
    --cpu-offload-gb 10 \
    --enforce-eager \
    --distributed-executor-backend ray
```

### Query the API

Call the openAI-like API started by the serve command

 - List avaialable models: `python -m llm.xplore api lsm
 - Generate from a prompt: `python -m llm.xplore api gen 'How are you?'
