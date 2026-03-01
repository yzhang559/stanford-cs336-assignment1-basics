#!/usr/bin/env python3
import os
import numpy as np
from tokenizer import Tokenizer


def tokenize_file(input_path, output_path, tokenizer):
    print(f"Processing {input_path}...")
    all_tokens = []

    with open(input_path, 'r', encoding='utf-8') as f:
        chunk = []
        for i, line in enumerate(f):
            chunk.append(line)
            if len(chunk) >= 1000:
                text = ''.join(chunk)
                tokens = tokenizer.encode(text)
                all_tokens.extend(tokens)
                chunk = []
                if (i + 1) % 100000 == 0:
                    print(f"  {i + 1} lines, {len(all_tokens)} tokens...")

        if chunk:
            text = ''.join(chunk)
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)

    token_array = np.array(all_tokens, dtype=np.uint16)
    np.save(output_path, token_array)
    print(f"Saved {len(token_array)} tokens to {output_path}")


if __name__ == '__main__':
    root = os.path.dirname(os.path.abspath(__file__))

    vocab_file = os.path.join(root, "tinystories_output/vocab.json")
    merges_file = os.path.join(root, "tinystories_output/merges.txt")

    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(vocab_file, merges_file, special_tokens=["<|endoftext|>"])

    train_input = os.path.join(root, "../data/TinyStoriesV2-GPT4-train.txt")
    train_output = os.path.join(root, "../data/TinyStoriesV2-GPT4-train.npy")

    valid_input = os.path.join(root, "../data/TinyStoriesV2-GPT4-valid.txt")
    valid_output = os.path.join(root, "../data/TinyStoriesV2-GPT4-valid.npy")

    tokenize_file(train_input, train_output, tokenizer)
    tokenize_file(valid_input, valid_output, tokenizer)

    # Example output:
    # Saved 541228515 tokens to data/TinyStoriesV2-GPT4-train.npy
    # Processing data/TinyStoriesV2-GPT4-valid.txt...
    #   100000 lines, 3470258 tokens...
    # Saved 5465865 tokens to data/TinyStoriesV2-GPT4-valid.npy
    # Done!

    print("Done!")
