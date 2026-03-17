"""
Byte Pair Encoding (BPE) Training Algorithm
============================================

BPE builds a vocabulary by iteratively merging the most frequent adjacent byte pairs.

Algorithm:
1. Pre-tokenize: Split text into words using regex pattern (GPT-2 style), count word frequencies
2. Initialize vocabulary with 256 single-byte tokens + special tokens
3. Repeat until vocab_size reached:
   a. Count frequencies of all adjacent byte pairs across all words
   b. Find the most frequent pair (tie-break: lexicographically larger pair wins)
   c. Merge that pair into a new token, add to vocabulary
   d. Update word representations and pair frequencies

Optimization - Lazy Heap for O(log n) max-finding:
- Must push on BOTH increase AND decrease to maintain correct tie-breaking order
"""

from concurrent.futures.process import ProcessPoolExecutor
from multiprocessing import cpu_count
import collections
import heapq
import regex as re
import time

assert re.__name__ == "regex"  # sanity check


class LexicographicMax:
    """Wrapper for heap tie-breaking: picks lexicographically larger pair."""

    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair


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


def _work_slice(path, start, end, special_tokens) -> collections.Counter[bytes]:
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return pre_tokenize(text, special_tokens)


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[
    dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = init_vocab(special_tokens)
    max_merge = vocab_size - len(vocab)

    special_tokens = set(special_tokens)
    merges: list[tuple[bytes, bytes]] = []
    w_counts: collections.Counter[bytes] = collections.Counter()
    num_worker = cpu_count()

    # read the file and split them into chunks
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_worker, '<|endoftext|>'.encode('utf8'))

    jobs = list(zip(boundaries[:-1], boundaries[1:]))
    print(f"Pre-tokenizing with {len(jobs)} workers...", flush=True)

    # multi process 1490 ms train_bpe(), pre_tokenize is not bottleneck
    with ProcessPoolExecutor(max_workers=min(num_worker, len(jobs))) as executor:
        futures = [executor.submit(_work_slice, input_path, start, end, special_tokens) for start, end in jobs]
        for fu in futures:
            w_counts.update(fu.result())
    # single process: pre_tokenize 1271 ms / 2185 ms 58% of train_bpe()
    # w_counts = pre_tokenize(text, special_tokens)
    print(f"Pre-tokenization done. Unique words: {len(w_counts)}", flush=True)

    print("Building w_freq...", flush=True)
    w_freq = {
        tuple(bytes([b]) for b in word): cnt for word, cnt in w_counts.items()
    }
    print(f"w_freq built. Size: {len(w_freq)}", flush=True)

    sp_token_tuple = {
        tuple(bytes([b]) for b in s.encode('utf-8')) for s in special_tokens
    }

    print("Building pair frequencies...", flush=True)
    pair2word = collections.defaultdict(set)
    p_freq, pair2word = get_pair_freq(w_freq, sp_token_tuple, pair2word)
    print(f"Initial pairs: {len(p_freq)}. Building heap...", flush=True)

    # Build max-heap: use negative count for max-heap behavior
    # Heap entries: (-count, LexicographicMax) - LexicographicMax reverses comparison for correct tie-breaking
    pair_heap = [(-cnt, LexicographicMax(pair)) for pair, cnt in p_freq.items()]
    heapq.heapify(pair_heap)
    print(f"Heap built. Starting {max_merge} merges...", flush=True)
    merge_start = time.time()

    for i in range(max_merge):
        if not pair_heap:
            break

        # Pop stale entries until we find a valid one
        while pair_heap:
            neg_cnt, max_pair = heapq.heappop(pair_heap)
            highest_pair = max_pair.pair
            current_cnt = p_freq.get(highest_pair, 0)
            if current_cnt == -neg_cnt and current_cnt > 0:
                break
            # Stale entry, skip it
        else:
            # Heap exhausted
            break

        new_token = highest_pair[0] + highest_pair[1]
        merges.append(highest_pair)
        vocab[len(vocab)] = new_token

        update_freq(p_freq, pair2word, highest_pair, w_freq, pair_heap, i)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - merge_start
            print(
                f"merge {i + 1}/{max_merge} | "
                f"pairs={len(p_freq)} | "
                f"words={len(w_freq)} | "
                f"time={elapsed:.1f}s",
                flush=True
            )

    return vocab, merges


def get_pair_freq(w_freq, special_tk, pair2word):
    freq = collections.defaultdict(int)
    for word, count in w_freq.items():
        if word in special_tk or len(word) < 2:
            continue

        for pair in zip(word[:-1], word[1:]):
            freq[pair] = freq.get(pair, 0) + count
            pair2word[pair].add(word)
    return freq, pair2word


def update_freq(p_freq, pair2word, highest_pair, w_freq, pair_heap, merge_idx=0):
    """
    Merge highest_pair in all words, update frequencies.
    Returns set of changed pairs to push to heap once at the end.
    """
    words_to_process = pair2word.pop(highest_pair)
    if len(words_to_process) > 100000:
        print(f"  merge {merge_idx}: processing {len(words_to_process)} words for pair {highest_pair}", flush=True)
    # Remove the merged pair from p_freq so stale check works
    del p_freq[highest_pair]

    merged_token = highest_pair[0] + highest_pair[1]
    changed_pairs = set()

    for old_word in words_to_process:
        if old_word not in w_freq:
            continue

        count = w_freq.pop(old_word)
        
        # Decrease freq for old pairs and unlink
        for i in range(len(old_word) - 1):
            pair = (old_word[i], old_word[i + 1])
            if pair == highest_pair:
                continue  # Already removed
            p_freq[pair] -= count
            changed_pairs.add(pair)
            if p_freq[pair] <= 0:
                del p_freq[pair]
            # Unlink old_word from pair2word
            pw = pair2word.get(pair)
            if pw:
                pw.discard(old_word)
                if not pw:
                    del pair2word[pair]

        # Merge the word
        new_word = []
        i = 0
        while i < len(old_word):
            if i < len(old_word) - 1 and old_word[i] == highest_pair[0] and old_word[i + 1] == highest_pair[1]:
                new_word.append(merged_token)
                i += 2
            else:
                new_word.append(old_word[i])
                i += 1
        new_word = tuple(new_word)
        
        w_freq[new_word] = w_freq.get(new_word, 0) + count

        if len(new_word) < 2:
            continue

        # Increase freq for new pairs
        for j in range(len(new_word) - 1):
            pair = (new_word[j], new_word[j + 1])
            p_freq[pair] = p_freq.get(pair, 0) + count
            pair2word[pair].add(new_word)
            changed_pairs.add(pair)

    # Push changed pairs to heap once at the end (not inside the loop)
    for pair in changed_pairs:
        if pair in p_freq and p_freq[pair] > 0:
            heapq.heappush(pair_heap, (-p_freq[pair], LexicographicMax(pair)))


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
