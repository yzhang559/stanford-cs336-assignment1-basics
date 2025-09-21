import sys
import pytest
from tests.memlimit import memory_limit

from tests.common import FIXTURES_PATH
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path, VOCAB_PATH, MERGES_PATH

IS_LINUX = sys.platform.startswith("linux")
IS_DARWIN = sys.platform == "darwin"

# Run these on Linux (rlimit) or mac (fallback). Skip on Windows.
skip_not_unix = pytest.mark.skipif(
    not (IS_LINUX or IS_DARWIN),
    reason="Memory tests supported on Linux/macOS only."
)

@skip_not_unix
def test_encode_iterable_memory_usage():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    # IMPORTANT on mac: don't accumulate results into a list,
    # or you'll measure caller memory, not just the streaming encoder.
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt", encoding="utf-8") as f:
        for _ in _encode_iterable(tokenizer, f):
            pass  # consume without storing

@skip_not_unix
@pytest.mark.xfail(reason="Tokenizer.encode is expected to exceed 1MB.")
def test_encode_memory_usage():
    """
    We expect this test to fail, since Tokenizer.encode reads the whole file.
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt", encoding="utf-8") as f:
        contents = f.read()
        _ = _encode(tokenizer, contents)

# ---- helpers under test guard ----

# Choose fallback method on mac:
#   "rss"   -> process peak RSS (stricter; may include other allocations)
#   "trace" -> Python allocations via tracemalloc (ignores native mallocs)
MAC_METHOD = "rss"  # or "trace"

@memory_limit(int(1e6), method=MAC_METHOD)  # 1 MB
def _encode_iterable(tokenizer, iterable):
    yield from tokenizer.encode_iterable(iterable)

@memory_limit(int(1e6), method=MAC_METHOD)  # 1 MB
def _encode(tokenizer, text: str):
    return tokenizer.encode(text)
