import sys
import regex as re
from typing import Iterable, Iterator

from cs336_basics.train_bpe import PAT, merge
from cs336_basics.utils import load_vocab, load_merges, HERE


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=list[str]):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = set(special_tokens) if special_tokens else set()
        self.encoder: dict[bytes, int] = {token: i for i, token in vocab.items()}
        self.bpe_rank: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(self.merges)}
        self.processed: dict[bytes, list[int]] = {}

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
        return list(self.encode_iterable([text]))
        #
        # if self.special_tokens:
        #     # skip the longer tokens first ie ["<docline>", "<doc"]
        #     special_sorted = sorted(self.special_tokens, key=len, reverse=True)
        #     # THE outer () is used to keep the special tokens
        #     split_keep = re.compile("(" + "|".join(re.escape(t) for t in special_sorted) + ")")
        #
        #     chunks = split_keep.split(text)
        #
        # else:
        #     chunks = [text]
        #
        # for chunk in chunks:
        #     if not chunk:
        #         continue
        #     if chunk in self.special_tokens:
        #         encoded.append(self.encoder.get(chunk.encode('utf-8')))
        #     else:
        #         for match in re.finditer(PAT, chunk):
        #             w_byte = match.group(0).encode('utf8')
        #             w_ids = self._encode_word_bytes(w_byte)
        #             encoded.extend(w_ids)
        # return encoded

    @staticmethod
    def _get_pair(w_bytes: list[bytes]):
        return set(zip(w_bytes[:-1], w_bytes[1:]))

    def _encode_word_bytes(self, w_byte: bytes) -> list[int]:
        # remove the unbounded cache
        # if w_byte in self.processed:
        #     return self.processed[w_byte]

        w_bytes = [w_byte[i:i + 1] for i in range(len(w_byte))]
        while True:
            pairs = self._get_pair(w_bytes)
            if not pairs:
                break

            best_pair = min(pairs, key=lambda pair: self.bpe_rank.get(pair, sys.maxsize))
            if self.bpe_rank.get(best_pair, sys.maxsize) == sys.maxsize:
                break

            w_bytes = merge(w_bytes, best_pair)

        result = []
        for item in w_bytes:
            # # TODO: Why key error?
            if item in self.encoder:
                result.append(self.encoder[item])

        # self.processed[w_byte] = result
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Stream-encode chunks of text, yielding token ids without building large lists.
        Matches the semantics of encode(text) but with O(1) working memory.
        """
        split_keep = None
        specials = self.special_tokens or []
        if specials:
            special_sorted = sorted(specials, key=len, reverse=True)
            split_keep = re.compile("(" + "|".join(re.escape(t) for t in special_sorted) + ")")

        for text_chunk in iterable:
            if not text_chunk:
                continue

            if split_keep:
                chunks = split_keep.split(text_chunk)
            else:
                chunks = [text_chunk]

            for chunk in chunks:
                if not chunk:
                    continue
                if chunk in specials:
                    tok_id = self.encoder.get(chunk.encode("utf-8"))
                    if tok_id is not None:
                        yield tok_id
                    continue

                for m in re.finditer(PAT, chunk):
                    w_byte = m.group(0).encode("utf-8")
                    for tid in self._encode_word_bytes(w_byte):
                        yield tid

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text
        """
        decoded = b"".join(self.vocab.get(i, b"") for i in ids)
        return decoded.decode("utf-8", errors="replace")


if __name__ == '__main__':
    t = Tokenizer.from_files(HERE / "tinystories_output" / "vocab.json", HERE / "tinystories_output" / "merges.txt",
                             special_tokens=["<|endoftext|>"])
    text = '''Once upon a time there was a little boy named Ben. Ben loved to explore the world around him. He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben was walking through the store when he came across a very special vase. When Ben saw it he was amazed!  
He said, “Wow, that is a really amazing vase! Can I buy it?” 
The shopkeeper smiled and said, “Of course you can. You can take it home and show all your friends how amazing it is!”
So Ben took the vase home and he was so proud of it! He called his friends over and showed them the amazing vase. All his friends thought the vase was beautiful and couldn't believe how lucky Ben was. 
And that's how Ben found an amazing vase in the store!
<|endoftext|>
Once upon a time, there was a reliable otter named Ollie. He lived in a river with his family. They all loved to play and swim together.
One day, Ollie's mom said, "Ollie, hurry and get some fish for dinner!" Ollie swam fast to catch fish. He saw his friend, the duck. "Hi, Ollie!" said the duck. "Hi, duck!" said Ollie. "I need to hurry and catch fish for my family."
While Ollie was catching fish, he found a big shiny stone. He thought, "This is not a fish, but it is so pretty!" Ollie took the shiny stone home to show his family. They all looked at the shiny stone and smiled. The shiny stone made everyone happy, and they forgot about the fish for dinner.
<|endoftext|>'''
    ids = t.encode(text)
    print(ids)
    print(t.decode(ids))
