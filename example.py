import os
import torch
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

# nsys profile -w true -t cuda,nvtx,cudnn,cublas   --capture-range=cudaProfilerApi   --capture-range-end=stop   --cuda-memory-usage=true   -o vllm_profile   python3 example.py


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    # start_cmd = f"nsys start --stats=true --session=vllm_test --trace=nvtx,cuda,cublas,cudnn,osrt --cuda-graph-trace=node --wait=all --cuda-memory-usage=true --trace-fork-before-exec=true --gpu-metrics-devices=all"
    # print(start_cmd)
    # os.system(start_cmd)
    torch.cuda.cudart().cudaProfilerStart()
    outputs = llm.generate(prompts, sampling_params)
    torch.cuda.cudart().cudaProfilerStop()
    # end_cmd = f"nsys stop --session=vllm_test"
    # print(end_cmd)
    # os.system(end_cmd)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
