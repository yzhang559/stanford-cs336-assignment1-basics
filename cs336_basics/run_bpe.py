from cs336_basics.train_bpe import train_bpe
from cs336_basics.utils import now, get_peak_rss_bytes, save_output, get_longest_token, HERE

TINY_STORY_DIR = "tinystories_output"
OPEN_WEB_DIR = "openweb_output"


def train_bpe_tinystories():
    start = now()
    vocab, merges = train_bpe(HERE.parent / "data/TinyStoriesV2-GPT4-train.txt", 10000,
                              special_tokens=["<|endoftext|>"])
    elapsed = now() - start
    print(f"time: {elapsed:.2f}s, peak RSS: {get_peak_rss_bytes() / 1024 / 1024:.2f} MB")

    save_output(vocab, merges, TINY_STORY_DIR)


# TODO: run it non-local, out of memory
def train_bpe_expts_owt():
    start = now()
    vocab, merges = train_bpe(HERE.parent / "data/owt_train.txt", 32000,
                              special_tokens=["<|endoftext|>"])
    elapsed = now() - start
    print(f"time: {elapsed:.2f}s, peak RSS: {get_peak_rss_bytes() / 1024 / 1024:.2f} MB")

    save_output(vocab, merges, OPEN_WEB_DIR)


if __name__ == '__main__':
    train_bpe_tinystories()
    get_longest_token(HERE / TINY_STORY_DIR / "vocab.json")
