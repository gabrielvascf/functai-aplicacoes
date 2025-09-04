from functai import configure, ai
from attachments import Attachments, set_verbose
import dspy
import ovllm
import threading
import subprocess

set_verbose(True)


def run_ollama_serve():
    subprocess.run(["ollama", "serve"])


thread = threading.Thread(target=run_ollama_serve)
lm = dspy.LM("ollama_chat/qwen3:4b", api_base="http://localhost:11434", api_key="")
thread.start()
configure(model="llama2", provider="ollama")
dspy.configure(lm=ovllm.llm)
configure(lm=lm, temperature=1.0)


def website_retriever(url):
    "Retrieves text content of a URL"
    return Attachments(url + "[images: false]").text


@ai
def rag(question) -> str: ...


print(rag("What is the square root of 2?"))
