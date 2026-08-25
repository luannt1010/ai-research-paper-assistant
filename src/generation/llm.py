from langchain_ollama import ChatOllama

class LLM:
    def __init__(self, model_name: str = "qwen3.8", temperature: float = 0, num_gpu: int = 1):
        self.llm = ChatOllama(model=model_name, temperature=temperature,
                              num_gpu=num_gpu, num_ctx=4096, top_p=0.9, reasoning=False,
                              top_k=40, num_predict=2048, validate_model_on_init=True)

    def inference(self, message):
        return self.llm.invoke(message)