# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os

def from_pretrained(cls, model_name, kwargs, cache_dir):
    # use local model if it exists
    local_path = os.path.join(cache_dir, 'local.' + model_name.replace("/", "_"))
    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir)

# predefined models
model_fullnames = {  'gpt2': 'gpt2',
                     'gpt2-xl': 'gpt2-xl',
                     'distilgpt2': 'distilbert/distilgpt2',
                     'gpt-neo-125m': 'EleutherAI/gpt-neo-125m',
                     'gpt-neo-1.3B': 'EleutherAI/gpt-neo-1.3B',
                     'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
                     'gpt-neox-20b': 'EleutherAI/gpt-neox-20b',
                     'gpt-j-6B': 'EleutherAI/gpt-j-6B',
                     'mgpt': 'sberbank-ai/mGPT',
                     'pubmedgpt': 'stanford-crfm/pubmedgpt',
                     'mt5-xl': 'google/mt5-xl',
                     'llama-13b': 'huggyllama/llama-13b',
                     'llama2-13b': 'TheBloke/Llama-2-13B-fp16',
                     'bloom-7b1': 'bigscience/bloom-7b1',
                     'opt-125m': 'facebook/opt-125m',
                     'opt-350m': 'facebook/opt-350m',
                     'opt-1.3b': 'facebook/opt-1.3b',
                     'opt-2.7b': 'facebook/opt-2.7b',
                     'opt-6.7b': 'facebook/opt-6.7b',
                     'opt-13b': 'facebook/opt-13b',
                     'pythia-70m': 'EleutherAI/pythia-70m',
                     'pythia-70m-dd': 'EleutherAI/pythia-70m-deduped',
                     'pythia-160m': 'EleutherAI/pythia-160m',
                     'pythia-160m-dd': 'EleutherAI/pythia-160m-deduped',
                     'pythia-410m': 'EleutherAI/pythia-410m',
                     'pythia-1b': 'EleutherAI/pythia-1b',
                     'pythia-1.4b': 'EleutherAI/pythia-1.4b',
                     'pythia-2.8b': 'EleutherAI/pythia-2.8b',
                     'pythia-6.9b': 'EleutherAI/pythia-6.9b',
                     'palmyra-small': 'Writer/palmyra-small',
                     'palmyra-base': 'Writer/palmyra-base',
                     }
float16_models = ['gpt-j-6B', 'gpt-neox-20b', 'llama-13b', 'llama2-13b', 'bloom-7b1', 'opt-6.7b', 'opt-13b', 'pythia-6.9b', 't5-11b']

def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name

def get_reference_model_name(name, ref_name):
    if 'pythia' in name and 'pythia' not in ref_name:
        return 'pythia-6.9b'

    if 'opt' in name and 'opt' not in ref_name:
        return 'opt-13b'

    if 'gpt' in name and 'gpt' not in ref_name:
        return 'gpt-j-6B'

    if 'palmyra' in name and 'palmyra' not in ref_name:
        return 'palmyra-base'

    return 'gpt-j-6B'

def load_model(model_name, device, cache_dir):
    model_fullname = get_model_fullname(model_name)
    print(f'Loading model {model_fullname}...')
    model_kwargs = {}
    if model_name in float16_models:
        model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gpt-j' in model_name:
        model_kwargs.update(dict(revision='float16'))
    model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
    model.to(device)
    print(f'DONE ({time.time() - start:.2f}s)')
    return model

def load_tokenizer(model_name, for_dataset, cache_dir):
    model_fullname = get_model_fullname(model_name)
    optional_tok_kwargs = {}
    if "facebook/opt-" in model_fullname:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if for_dataset in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    else:
        optional_tok_kwargs['padding_side'] = 'right'
    base_tokenizer = from_pretrained(AutoTokenizer, model_fullname, optional_tok_kwargs, cache_dir=cache_dir)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        if '13b' in model_fullname:
            base_tokenizer.pad_token_id = 0
    return base_tokenizer


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="bloom-7b1")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    load_tokenizer(args.model_name, 'xsum', args.cache_dir)
    load_model(args.model_name, 'cpu', args.cache_dir)
