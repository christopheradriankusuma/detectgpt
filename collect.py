from openai import OpenAI
from datasets import load_dataset
import pandas as pd
from dotenv import dotenv_values

import tiktoken

config = dotenv_values(".env")

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

client = OpenAI(
    api_key=config['API_KEY'],
    organization=config['ORG'],
)

# links = [
#     'euclaise/writingprompts', # story
#     'squad', # context
#     'EdinburghNLP/xsum', # document
#     'pubmed_qa', # question, long_answer
#     'wmt16', # de-en subset
# ]

def complete(sentence, max_tokens=200, temperature=1, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        # logprobs=True,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "user", "content": sentence},
        ]
    )
    return response.choices[0].message.content

def squad(model="gpt-3.5-turbo"):
    df = pd.DataFrame(columns=['human', 'machine'])
    ds = load_dataset('squad', split='train[:10000]')
    contexts = list(set(ds['context']))
    index = 0
    for context in contexts:
        if index == 500:
            break

        trimmed = ' '.join(context.split()[:30])
        nc = num_tokens_from_string(context)
        nt = num_tokens_from_string(trimmed)

        if nc - nt < 30:
            continue

        completed = complete(trimmed, max_tokens=nc-nt, model=model)
        full = trimmed.strip() + ' ' + completed.strip()

        # print(nc, nt, num_tokens_from_string(full))
        df.loc[index] = [context, full]
        index += 1

    print(f'generated {index} wikipedia/squad examples')
    df.to_csv('squad-hc.csv', index=False)

def writingprompts(model="gpt-3.5-turbo"):
    df = pd.DataFrame(columns=['human', 'machine'])
    ds = load_dataset('euclaise/writingprompts', split='train[:10000]')
    stories = list(set(ds['story']))
    index = 0
    for story in stories:
        if index == 500:
            break

        trimmed = ' '.join(story.split()[:30])
        nc = num_tokens_from_string(story)
        nt = num_tokens_from_string(trimmed)

        if nc - nt < 30:
            continue

        completed = complete(trimmed, max_tokens=nc-nt, model=model)
        full = trimmed.strip() + ' ' + completed.strip()

        # print(nc, nt, num_tokens_from_string(full))
        df.loc[index] = [story, full]
        index += 1

    print(f'generated {index} writingprompts examples')
    df.to_csv('writingprompts-hc.csv', index=False)

def xsum(model="gpt-3.5-turbo"):
    df = pd.DataFrame(columns=['human', 'machine'])
    ds = load_dataset('EdinburghNLP/xsum', split='train[:10000]')
    documents = list(set(ds['document']))
    index = 0
    for document in documents:
        if index == 500:
            break

        trimmed = ' '.join(document.split()[:30])
        nc = num_tokens_from_string(document)
        nt = num_tokens_from_string(trimmed)

        if nc - nt < 30:
            continue

        completed = complete(trimmed, max_tokens=nc-nt, model=model)
        full = trimmed.strip() + ' ' + completed.strip()

        # print(nc, nt, num_tokens_from_string(full))
        df.loc[index] = [document, full]
        index += 1

    print(f'generated {index} xsum examples')
    df.to_csv('xsum-hc.csv', index=False)

if __name__ == "__main__":
    pass
    # squad()
    # writingprompts()
    # xsum()
