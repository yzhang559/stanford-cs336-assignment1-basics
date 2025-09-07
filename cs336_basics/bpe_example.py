from cs336_basics.pretokenization_example import find_chunk_boundaries, HERE

WS = {9, 10, 11, 12, 13, 32}  # \t \n \v \f \r ' '
EOF = '<|endoftext|>'


def toy_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[
    dict[int, bytes], list[tuple[bytes, bytes]]]:
    merges: dict[tuple[bytes, bytes], int] = {}

    vocab = init_vocab()
    for i, stoken in enumerate(special_tokens):
        vocab[256 + i] = stoken.encode('utf-8')

    base_vocab_size = len(vocab)
    merge_cnt = vocab_size - base_vocab_size

    ids: list[int] = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 1, EOF.encode('utf-8'))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            tokens = pre_tokenize(f.read(end - start))
            ids.extend(list(tokens))

            for i in range(merge_cnt):
                freq = get_freq(ids)
                if not freq:
                    break
                _, pair = get_most_frequent_pair(freq)
                new_id = base_vocab_size + i

                ids = merge(ids, new_id, pair)
                merges[pair] = new_id
                vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
                print(f'merge {vocab[pair[0]]} and {vocab[pair[1]]} : {vocab[new_id]}')

    return vocab, list(tuple(merges.keys()))


def init_vocab() -> dict[int, bytes]:
    return {i: bytes([i]) for i in range(256)}


def pre_tokenize(raw_tokens):
    return raw_tokens


def get_most_frequent_pair(freq):
    pair, count = max(freq.items(), key=lambda kv: (kv[1], kv[0]))
    return count, pair


def get_freq(ids):
    freq = {}
    for a, b in zip(ids, ids[1:]):
        # should not count across ASCII white space
        if a in WS or b in WS:
            continue
        freq[(a, b)] = freq.get((a, b), 0) + 1

    return freq


def merge(ids, idx, pair):
    updated_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            updated_ids.append(idx)
            i += 2
        else:
            updated_ids.append(ids[i])
            i += 1
    return updated_ids


if __name__ == '__main__':
    v, m = toy_bpe(HERE / "corpus.txt", 256 + 1 + 6, special_tokens=[EOF])
    print(v)
