import sys
from typing import Iterable, Iterator

from cs336_basics.utils import load_vocab, load_merges, HERE


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=list[str]):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens. This method should accept the following additional parameters:
        """
        vocab = load_vocab(vocab_filepath)
        merges = load_merges(merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-eﬀicient tokenization of large files that we cannot directly load into
        memory.
        """
        pass

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text
        """

if __name__ == '__main__':
    t = Tokenizer.from_files(HERE / "tinystories_output" / "vocab.json", HERE / "tinystories_output" / "merges.txt")
    print(t.merges)
    print(t.vocab)
