#!/usr/bin/env python3
import argparse
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

    parser = argparse.ArgumentParser(description="Tokenize TinyStories text files into .npy token ID arrays")
    parser.add_argument(
        "--vocab-file",
        default=os.path.join(root, "tinystories_output/vocab.json"),
        help="Path to vocab.json",
    )
    parser.add_argument(
        "--merges-file",
        default=os.path.join(root, "tinystories_output/merges.txt"),
        help="Path to merges.txt",
    )
    parser.add_argument(
        "--train-input",
        default=os.path.join(root, "../data/TinyStoriesV2-GPT4-train.txt"),
        help="Path to training .txt file",
    )
    parser.add_argument(
        "--train-output",
        default=os.path.join(root, "../data/TinyStoriesV2-GPT4-train.npy"),
        help="Path to output training .npy file",
    )
    parser.add_argument(
        "--valid-input",
        default=os.path.join(root, "../data/TinyStoriesV2-GPT4-valid.txt"),
        help="Path to validation .txt file",
    )
    parser.add_argument(
        "--valid-output",
        default=os.path.join(root, "../data/TinyStoriesV2-GPT4-valid.npy"),
        help="Path to output validation .npy file",
    )
    args = parser.parse_args()

    vocab_file = args.vocab_file
    merges_file = args.merges_file

    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(vocab_file, merges_file, special_tokens=["<|endoftext|>"])

    train_input = args.train_input
    train_output = args.train_output

    valid_input = args.valid_input
    valid_output = args.valid_output

    tokenize_file(train_input, train_output, tokenizer)
    tokenize_file(valid_input, valid_output, tokenizer)

    # Example output:
    # Saved 541228515 tokens to data/TinyStoriesV2-GPT4-train.npy
    # Processing data/TinyStoriesV2-GPT4-valid.txt...
    #   100000 lines, 3470258 tokens...
    # Saved 5465865 tokens to data/TinyStoriesV2-GPT4-valid.npy
    # Done!

    print("Done!")
