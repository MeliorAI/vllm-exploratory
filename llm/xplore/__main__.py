import click

from openai import OpenAI

from llm.xplore import DEFAULT_MODEL
from llm.xplore.batch import BatchLLM

# Set OpenAI's API key and API base to use vLLM's API server.
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8000/v1"


@click.group()
def cli():
    """Main CLI to interact with your LLM"""


@click.group("api")
def api():
    """CLI to interact with your LLM via an openAI-like API"""


@click.group("server")
def server():
    """CLI to run your LLM via vLLM's server"""


@click.group("batch")
def batch():
    """CLI to interact with your LLM off-line in batch-mode"""


@server.command("run")
@click.option("-m", "model", default=DEFAULT_MODEL)
@click.option("-t", "--template", default="templates/template_chatml.jinja")
@click.option("-l", "--max-model-len", default=300)
@click.option("-c", "--cpu-offload-gb", default=10)
@click.option("-e", "--enforce-eager", is_flag=True)
def server_run(model:str, template:str, max_model_len: int, cpu_offload_gb:int, enforce_eager:bool):
    import subprocess
    proc = subprocess.Popen(
        [
            "vllm",
            "serve",
            model,
            f"--chat-template={template}",
            f"--max-model-len={max_model_len}",
            f"--cpu-offload-gb={cpu_offload_gb}",
            "--enforce-eager",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE
    )
    # Capture output in real-time
    for line in iter(proc.stdout.readline, b""):
        print(line.decode(), end="")
    # Wait for the process to complete
    proc.wait()

    # Get the exit status
    exit_code = proc.returncode
    print(f"Process exited with code: {exit_code}")

@batch.command("run")
@click.option("-m", "model", default=DEFAULT_MODEL)
@click.option("-p", "--prompt", multiple=True)
@click.option("-i", "--input-file")
def batch_run(model:str, prompt: list[str] = [], input_file:str | None = None):
    if prompt:
        print(f"Received {len(prompt)} prompts")
        inputs = list(prompt)
    elif input_file:
        with open(input_file, "r") as f:
            inputs = [l.strip() for l in f.readlines()]
    else:
        raise ValueError("At least one prompt OR input-file is required")

    llm = BatchLLM(model)
    llm.generate(prompts=inputs)


@api.command("lsm")
@click.option("-u", "--base-url", default=OPENAI_API_BASE)
@click.option("-k", "--api-key", default=OPENAI_API_KEY)
def list_models(base_url:str, api_key:str):
    """List available models in the server

    NOTE: Requires a running VLLM server

    Args:
        base_url (str): API base URL
        api_key (str): API key to authenticate with (if applicable)
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    res = client.models.list()

    print("Available models:")
    for m in res.data:
        print(f" - {m.id}")


@api.command("gen")
@click.argument("prompt")
@click.option("-m", "--model", default=None)
@click.option("-u", "--base-url", default=OPENAI_API_BASE)
@click.option("-k", "--api-key", default=OPENAI_API_KEY)
def completions_api(prompt:str, model:str | None, base_url:str, api_key:str):
    """Runs the completions API endpoint on the given model & prompt

    NOTE: Requires a running VLLM server

    Args:
        prompt (str): Text input to send to the model
        model (str): Which model to use, if not provided 1st from the server
        base_url (str): API base URL
        api_key (str): API key to authenticate with (if applicable)
    """
    print(f"☎ Calling API @ '{base_url}'...")

    # Then we can call the server as we would do with the openAI API client
    client = OpenAI(api_key=api_key, base_url=base_url)

    model_id = model or client.models.list().data[0].id
    chat_response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    print("Chat response:", chat_response.choices[0].message.content)


if __name__ == "__main__":
    cli.add_command(api)
    cli.add_command(batch)
    cli.add_command(server)
    cli()
