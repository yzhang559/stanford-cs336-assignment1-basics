#!/usr/bin/env python3
import argparse
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tokenizer import Tokenizer


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Tokenize text files into .npy token ID arrays using shard-to-disk processing."
    )

    parser.add_argument(
        "--vocab-file",
        default=str(root / "tinystories_output" / "vocab.json"),
        help="Path to vocab.json",
    )
    parser.add_argument(
        "--merges-file",
        default=str(root / "tinystories_output" / "merges.txt"),
        help="Path to merges.txt",
    )

    parser.add_argument(
        "--train-input",
        default=str(root.parent / "data" / "TinyStoriesV2-GPT4-train.txt"),
        help="Path to training .txt file",
    )
    parser.add_argument(
        "--train-output",
        default=str(root.parent / "data" / "TinyStoriesV2-GPT4-train.npy"),
        help="Path to output training .npy file",
    )

    parser.add_argument(
        "--valid-input",
        default=str(root.parent / "data" / "TinyStoriesV2-GPT4-valid.txt"),
        help="Path to validation .txt file",
    )
    parser.add_argument(
        "--valid-output",
        default=str(root.parent / "data" / "TinyStoriesV2-GPT4-valid.npy"),
        help="Path to output validation .npy file",
    )

    parser.add_argument(
        "--input-path",
        default=None,
        help="Optional single input file to tokenize instead of train/valid pair",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional single output .npy path used with --input-path",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of tokenizer worker processes",
    )
    parser.add_argument(
        "--lines-per-chunk",
        type=int,
        default=50000,
        help="How many lines to bundle into each tokenization chunk",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        default=["<|endoftext|>"],
        help="Special token(s) passed to Tokenizer.from_files. Repeatable.",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep intermediate shard files for debugging",
    )

    return parser


def chunk_text_file(input_path: str, chunk_dir: Path, lines_per_chunk: int) -> list[Path]:
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths: list[Path] = []
    total_lines = 0
    chunk_idx = 0
    buf: list[str] = []

    print(f"Chunking {input_path} with lines_per_chunk={lines_per_chunk}...", flush=True)

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            buf.append(line)
            total_lines += 1

            if len(buf) >= lines_per_chunk:
                chunk_path = chunk_dir / f"chunk_{chunk_idx:06d}.txt"
                with open(chunk_path, "w", encoding="utf-8") as out:
                    out.write("".join(buf))
                chunk_paths.append(chunk_path)
                buf.clear()
                chunk_idx += 1

                if chunk_idx % 20 == 0:
                    print(f"  wrote {chunk_idx} chunks, {total_lines} lines so far...", flush=True)

        if buf:
            chunk_path = chunk_dir / f"chunk_{chunk_idx:06d}.txt"
            with open(chunk_path, "w", encoding="utf-8") as out:
                out.write("".join(buf))
            chunk_paths.append(chunk_path)

    print(f"Prepared {len(chunk_paths)} chunks from {total_lines} lines.", flush=True)
    return chunk_paths

def _worker_entry(args):
    return _tokenize_chunk_worker(*args)

def _tokenize_chunk_worker(
    chunk_path: str,
    shard_path: str,
    vocab_file: str,
    merges_file: str,
    special_tokens: list[str],
) -> tuple[str, int]:
    tokenizer = Tokenizer.from_files(
        vocab_file,
        merges_file,
        special_tokens=special_tokens,
    )

    with open(chunk_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    arr = np.asarray(tokens, dtype=np.uint16)
    arr.tofile(shard_path)

    return shard_path, int(arr.size)


def tokenize_to_shards(
    input_path: str,
    vocab_file: str,
    merges_file: str,
    shard_dir: Path,
    num_workers: int,
    lines_per_chunk: int,
    special_tokens: list[str],
) -> tuple[list[Path], int]:
    chunk_dir = shard_dir / "chunks"
    bin_dir = shard_dir / "bins"
    chunk_paths = chunk_text_file(input_path, chunk_dir, lines_per_chunk)
    bin_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for i, chunk_path in enumerate(chunk_paths):
        shard_path = bin_dir / f"part_{i:06d}.bin"
        jobs.append(
            (
                str(chunk_path),
                str(shard_path),
                vocab_file,
                merges_file,
                special_tokens,
            )
        )

    total_tokens = 0
    shard_paths: list[Path] = []

    print(f"Tokenizing with {num_workers} worker(s)...", flush=True)

    if num_workers == 1:
        for i, job in enumerate(jobs, 1):
            shard_path, token_count = _tokenize_chunk_worker(*job)
            shard_paths.append(Path(shard_path))
            total_tokens += token_count
            print(
                f"  chunk {i}/{len(jobs)} | shard={Path(shard_path).name} | "
                f"chunk_tokens={token_count} | total_tokens={total_tokens}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i, (shard_path, token_count) in enumerate(executor.map(_worker_entry, jobs), 1):
                shard_paths.append(Path(shard_path))
                total_tokens += token_count
                print(
                    f"  chunk {i}/{len(jobs)} | shard={Path(shard_path).name} | "
                    f"chunk_tokens={token_count} | total_tokens={total_tokens}",
                    flush=True,
                )

    return shard_paths, total_tokens


def merge_shards_to_npy(shard_paths: list[Path], output_path: str, total_tokens: int) -> None:
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Merging {len(shard_paths)} shard(s) into {output_path}...", flush=True)

    arr = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.uint16,
        shape=(total_tokens,),
    )

    offset = 0
    for i, shard_path in enumerate(sorted(shard_paths), 1):
        shard = np.fromfile(shard_path, dtype=np.uint16)
        n = shard.size
        arr[offset : offset + n] = shard
        offset += n

        print(
            f"  merged {i}/{len(shard_paths)} | {shard_path.name} | "
            f"shard_tokens={n} | written={offset}/{total_tokens}",
            flush=True,
        )

    arr.flush()
    print(f"Saved {total_tokens} tokens to {output_path}", flush=True)


def tokenize_file(
    input_path: str,
    output_path: str,
    vocab_file: str,
    merges_file: str,
    num_workers: int,
    lines_per_chunk: int,
    special_tokens: list[str],
    keep_shards: bool,
) -> None:
    print(f"Processing input={input_path}", flush=True)
    print(f"Output will be saved to {output_path}", flush=True)

    shard_dir = Path(tempfile.mkdtemp(prefix="tokenize_shards_"))

    try:
        shard_paths, total_tokens = tokenize_to_shards(
            input_path=input_path,
            vocab_file=vocab_file,
            merges_file=merges_file,
            shard_dir=shard_dir,
            num_workers=num_workers,
            lines_per_chunk=lines_per_chunk,
            special_tokens=special_tokens,
        )
        merge_shards_to_npy(shard_paths, output_path, total_tokens)
    finally:
        if keep_shards:
            print(f"Keeping intermediate files at: {shard_dir}", flush=True)
        else:
            shutil.rmtree(shard_dir, ignore_errors=True)


def main() -> None:
    args = build_parser().parse_args()

    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.lines_per_chunk < 1:
        raise ValueError("--lines-per-chunk must be >= 1")

    if args.input_path is not None or args.output_path is not None:
        if not args.input_path or not args.output_path:
            raise ValueError("--input-path and --output-path must be provided together")

        tokenize_file(
            input_path=args.input_path,
            output_path=args.output_path,
            vocab_file=args.vocab_file,
            merges_file=args.merges_file,
            num_workers=args.num_workers,
            lines_per_chunk=args.lines_per_chunk,
            special_tokens=args.special_token,
            keep_shards=args.keep_shards,
        )
    else:
        tokenize_file(
            input_path=args.train_input,
            output_path=args.train_output,
            vocab_file=args.vocab_file,
            merges_file=args.merges_file,
            num_workers=args.num_workers,
            lines_per_chunk=args.lines_per_chunk,
            special_tokens=args.special_token,
            keep_shards=args.keep_shards,
        )
        tokenize_file(
            input_path=args.valid_input,
            output_path=args.valid_output,
            vocab_file=args.vocab_file,
            merges_file=args.merges_file,
            num_workers=args.num_workers,
            lines_per_chunk=args.lines_per_chunk,
            special_tokens=args.special_token,
            keep_shards=args.keep_shards,
        )

    print("Done!", flush=True)


if __name__ == "__main__":
    main()