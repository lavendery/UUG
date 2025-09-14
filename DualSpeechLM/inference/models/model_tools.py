import torch
from .llama_xformer import LlamaForCausalLM
from .llama_xformer import GPTLlama
from .modeling_phi3 import GPTPhi3ForCausalLM


def get_pretrained_llama_causal_model(pretrained_model_name_or_path=None, torch_dtype='fp16', **kwargs):
    if torch_dtype == 'fp16' or torch_dtype == 'float16':
        torch_dtype = torch.float16
    elif torch_dtype == 'bf16' or torch_dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    else:
        torch_dtype == torch.float32
    # model = LlamaForCausalLM.from_pretrained(
    model = GPTLlama.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        **kwargs,
    )

    return model

def get_pretrained_phi35mini_instruct_model(pretrained_model_name_or_path=None, torch_dtype='fp16', **kwargs):
    if torch_dtype == 'fp16' or torch_dtype == 'float16':
        torch_dtype = torch.float16
    elif torch_dtype == 'bf16' or torch_dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    else:
        torch_dtype == torch.float32
    model = GPTPhi3ForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        **kwargs,
    )

    return model
