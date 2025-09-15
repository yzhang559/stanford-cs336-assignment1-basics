import collections
import regex as re
assert re.__name__ == "regex"  # sanity check

from cs336_basics.pretokenization_example import find_chunk_boundaries, HERE

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def init_vocab(sp_tokens) -> dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}
    for i, sp in enumerate(sp_tokens):
        vocab[i + 256] = sp.encode('utf8')
    return vocab


def pre_tokenize(text: str, special_tokens: set[str]) -> collections.Counter[bytes]:
    freq_table = collections.Counter()
    if special_tokens:
        # skip the longer tokens first ie ["<docline>", "<doc"]
        special_sorted = sorted(special_tokens, key=len, reverse=True)
        # THE outer () is used to keep the special tokens
        split_keep = re.compile("(" + "|".join(re.escape(t) for t in special_sorted) + ")")

        chunks = split_keep.split(text)

    else:
        chunks = [text]

    for chunk in chunks:
        if not chunk:
            continue
        if chunk in special_tokens:
            w_byte = chunk.encode('utf8')
            freq_table[w_byte] += 1
        else:
            for match in re.finditer(PAT, chunk):
                w_byte = match.group(0).encode('utf8')
                freq_table[w_byte] += 1

    return freq_table


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[
    dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = init_vocab(special_tokens)
    max_merge = vocab_size - len(vocab)

    special_tokens = set(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    # read the file and split them into chunks
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 1, '<|endoftext|>'.encode('utf8'))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            text = f.read(end - start).decode("utf-8", errors="ignore")

    w_counts = pre_tokenize(text, special_tokens)

    w_freq = {
        tuple(bytes([b]) for b in word): cnt for word, cnt in w_counts.items()
    }

    sp_token_tuple = {
        tuple(bytes([b]) for b in s.encode('utf-8')) for s in special_tokens
    }

    pair2word = collections.defaultdict(set)
    p_freq, pair2word = get_pair_freq(w_freq, sp_token_tuple, pair2word)

    for i in range(max_merge):
        if not p_freq:
            break

        highest_pair = get_most_frequent_pair(p_freq)

        new_token = highest_pair[0] + highest_pair[1]
        merges.append(highest_pair)
        vocab[len(vocab)] = new_token

        update_freq(p_freq, pair2word, highest_pair, w_freq)

    return vocab, merges


def get_most_frequent_pair(freq):
    return max(freq, key=lambda p: (freq[p], p))


def get_pair_freq(w_freq, special_tk, pair2word):
    freq = collections.defaultdict(int)
    for word, count in w_freq.items():
        if word in special_tk or len(word) < 2:
            continue

        for pair in zip(word[:-1], word[1:]):
            freq[pair] = freq.get(pair, 0) + count
            pair2word[pair].add(word)
    return freq, pair2word


def update_freq(p_freq, pair2word, highest_pair, w_freq):
    to_processed = list(pair2word.pop(highest_pair))
    for word in to_processed:
        if word not in w_freq:
            continue

        count = w_freq.pop(word)
        for pair in zip(word[:-1], word[1:]):
            if pair in p_freq:
                p_freq[pair] = p_freq.get(pair, 0) - count
                if p_freq[pair] == 0:
                    del p_freq[pair]
        for pair in set(zip(word[:-1], word[1:])):
            if pair in pair2word:
                pair2word[pair].discard(word)
                if not pair2word[pair]:
                    del pair2word[pair]

        new_word = merge(word, highest_pair)
        w_freq[new_word] = w_freq.get(new_word, 0) + count

        if len(new_word) < 2:
            continue

        for pair in zip(new_word[:-1], new_word[1:]):
            p_freq[pair] += count
            pair2word[pair].add(new_word)


def merge(w, pair) -> tuple[bytes]:
    updated_word = []
    i = 0
    while i < len(w):
        if i < len(w) - 1 and w[i] == pair[0] and w[i + 1] == pair[1]:
            updated_word.append(pair[0] + pair[1])
            i += 2
        else:
            updated_word.append(w[i])
            i += 1
    return tuple(updated_word)


if __name__ == '__main__':
    vocab, merges = train_bpe(HERE / "corpus.txt", 256 + 6, special_tokens=["<|endoftext|>"])
    print(vocab)
    print(merges)
