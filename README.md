# LLM-exploratory


# HowTo

## Serve

Run a vLLM server:

```shell
# Example vllm server command:
# To see all options check 'vllm serve --help'
vllm serve <modelID> \
    --chat-template templates/template_chatml.jinja \
    --device auto \
    --max-model-len 400 \
    --cpu-offload-gb 10 \
    --enforce-eager \
    --distributed-executor-backend ray
```

## TBD
