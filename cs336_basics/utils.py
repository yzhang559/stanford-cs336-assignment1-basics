import base64
import json
import time, os, sys
from pathlib import Path

VOCAB_FILE = "vocab.json"
MERGES_FILE = "merges.txt"
HERE = Path(__file__).resolve().parent


def _b64encode(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _b64decode(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def now():
    return time.perf_counter()


def get_peak_rss_bytes():
    # Cross-platform best-effort:
    # - Linux: resource.ru_maxrss is in kilobytes
    # - macOS: ru_maxrss is in bytes
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        peak = rusage.ru_maxrss
        # Heuristic: assume KB on Linux (most common), bytes on macOS
        if sys.platform == "darwin":
            return int(peak)  # bytes
        else:
            return int(peak) * 1024  # KB -> bytes
    except Exception:
        # Fallback to psutil current RSS (not true peak, but better than nothing)
        try:
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss
        except Exception:
            return -1


def save_output(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], output_dir: str):
    serializable_vocab = {k: list(v) for k, v in vocab.items()}
    with open(HERE / output_dir / VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)
    print("Wrote vocabulary to", output_dir)

    with open(HERE / output_dir / MERGES_FILE, "w", encoding="utf-8") as f:
        for a, b in merges:
            rec = {"a": _b64encode(a), "b": _b64encode(b)}
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    print("Wrote merges to", output_dir)


def get_longest_token(input_path):
    data = json.load(open(input_path, "r", encoding="utf-8"))
    tokens = list(data.items())
    tokens.sort(key=lambda kv: len(kv[1]), reverse=True)

    raw = bytes(tokens[0][1])
    longest_token = raw.decode("utf-8", errors="ignore")
    print(f"Longest token: {longest_token}, length: {len(raw)}, raw bytes: {raw}")
    return longest_token


def load_vocab(vocab_file) -> dict[int, bytes]:
    data = json.load(open(vocab_file, "r", encoding="utf-8"))
    deserializable_vocab = {k: bytes(v) for k, v in data.items()}
    return deserializable_vocab


def load_merges(merges_file) -> list[tuple[bytes, bytes]]:
    merges = []
    with open(merges_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            # use base64 to guarantee no ambiguity
            merges.append((_b64decode(rec["a"]), _b64decode(rec["b"])))
    return merges
