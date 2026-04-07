#!/usr/bin/env python3
import os
import shutil
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from requests.exceptions import ChunkedEncodingError
from urllib3.exceptions import ProtocolError

REPO_ROOT = Path('/home/bkai/cwq/MASC-CSC')
LLM_DIR = REPO_ROOT / 'LLM'
LLM_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ('Qwen/Qwen2.5-0.5B-Instruct', 'Qwen2.5-0.5B-Instruct'),
    ('Qwen/Qwen2.5-1.5B-Instruct', 'Qwen2.5-1.5B-Instruct'),
    ('Qwen/Qwen2.5-3B-Instruct', 'Qwen2.5-3B-Instruct'),
]
ALLOW_PATTERNS = [
    '*.json',
    '*.safetensors',
    '*.safetensors.index.json',
    '*.model',
    '*.py',
    'merges.txt',
    'vocab.json',
    'tokenizer*',
]
RESERVED_BYTES = 2 * 1024**3
MAX_RETRIES = 8
RETRY_BASE_SECONDS = 15


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def model_size(repo_id: str) -> int:
    api = HfApi()
    info = api.model_info(repo_id, files_metadata=True)
    total = 0
    for s in info.siblings:
        size = getattr(s, 'size', None)
        if isinstance(size, int):
            total += size
    return total


def ensure_link(snapshot: str, link_name: str) -> None:
    link_path = LLM_DIR / link_name
    if link_path.is_symlink() or link_path.is_file():
        link_path.unlink()
    elif link_path.is_dir():
        raise RuntimeError(f'refuse to overwrite existing directory: {link_path}')
    os.symlink(snapshot, link_path)


def download_with_retry(repo_id: str) -> str:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f'[download] {repo_id}: attempt {attempt}/{MAX_RETRIES}', flush=True)
            return snapshot_download(
                repo_id=repo_id,
                allow_patterns=ALLOW_PATTERNS,
                max_workers=1,
                resume_download=True,
            )
        except (ChunkedEncodingError, ProtocolError, OSError, TimeoutError) as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            sleep_s = RETRY_BASE_SECONDS * attempt
            print(
                f'[retry] {repo_id}: {type(exc).__name__}: {exc}. sleep {sleep_s}s then resume',
                flush=True,
            )
            time.sleep(sleep_s)
    raise RuntimeError(f'failed to download {repo_id} after {MAX_RETRIES} attempts: {last_error}')


def main() -> int:
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('PYTHONUNBUFFERED', '1')

    print(f'[info] LLM dir: {LLM_DIR}', flush=True)
    print(f'[info] free space before download: {free_bytes(REPO_ROOT) / 1024**3:.2f} GiB', flush=True)

    for repo_id, link_name in MODELS:
        target = LLM_DIR / link_name
        if target.is_symlink() and target.exists():
            print(f'[skip] {repo_id} already linked at {target}', flush=True)
            continue

        size = model_size(repo_id)
        free = free_bytes(REPO_ROOT)
        need = size + RESERVED_BYTES
        print(f'[plan] {repo_id}: estimated {size / 1024**3:.2f} GiB, free {free / 1024**3:.2f} GiB', flush=True)
        if free < need:
            print(f'[stop] not enough free space for {repo_id}; need about {need / 1024**3:.2f} GiB including reserve', flush=True)
            return 2

        print(f'[start] {repo_id}', flush=True)
        snapshot = download_with_retry(repo_id)
        ensure_link(snapshot, link_name)
        print(f'[done] {repo_id} -> {target} -> {snapshot}', flush=True)
        print(f'[info] free space now: {free_bytes(REPO_ROOT) / 1024**3:.2f} GiB', flush=True)

    print('[all_done] all planned Qwen models are available under LLM/', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
