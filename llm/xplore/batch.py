from vllm import LLM, SamplingParams

from llm.xplore import DEFAULT_MODEL


PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


class BatchLLM():
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        print(f"✨ Initializing LLM ({model})")
        self.sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        self.llm = LLM(model=model)


    def generate(self, prompts: list[str] = PROMPTS):
        outputs = self.llm.generate(prompts, self.sampling_params)

        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"🦜 Prompt: {prompt!r}, Generated text: {generated_text!r}")
