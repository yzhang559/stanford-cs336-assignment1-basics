from cs336_basics.train_bpe import train_bpe
from cs336_basics.utils import now, get_peak_rss_bytes, save_output, get_longest_token, HERE
import argparse

TINY_STORY_DIR = "tinystories_output"
OPEN_WEB_DIR = "openweb_output"


def train_bpe_tinystories():
    start = now()
    vocab, merges = train_bpe(HERE.parent / "data/TinyStoriesV2-GPT4-train.txt", 10000,
                              special_tokens=["<|endoftext|>"])
    elapsed = now() - start
    print(f"time: {elapsed:.2f}s, peak RSS: {get_peak_rss_bytes() / 1024 / 1024:.2f} MB")

    save_output(vocab, merges, TINY_STORY_DIR)


def train_bpe_expts_owt(input_path, vocab_size=32000):
    start = now()
    vocab, merges = train_bpe(input_path, vocab_size,
                              special_tokens=["<|endoftext|>"])
    elapsed = now() - start
    print(f"time: {elapsed:.2f}s, peak RSS: {get_peak_rss_bytes() / 1024 / 1024:.2f} MB")

    save_output(vocab, merges, OPEN_WEB_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BPE on input data')
    parser.add_argument('input_path', type=str, 
                       help='Path to input training data file')
    parser.add_argument('--vocab-size', type=int, default=32000,
                       help='Vocabulary size (default: 32000)')
    
    args = parser.parse_args()
    
    # train_bpe_tinystories()
    train_bpe_expts_owt(args.input_path, args.vocab_size)
    get_longest_token(HERE / OPEN_WEB_DIR / "vocab.json")
