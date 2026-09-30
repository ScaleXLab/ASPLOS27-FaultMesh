#!/usr/bin/env python3
"""Resumable multi-connection HTTP range download.

  python3 parallel_download.py <url> <out> <size_bytes> [workers]

Chunks already recorded in <out>.done are skipped on restart.
"""
import os
import subprocess
import sys
import threading

CHUNK = 256 * 1024 * 1024

url, out, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
done_path = out + '.done'

if not os.path.exists(out):
    with open(out, 'wb') as f:
        f.truncate(size)
done = set()
if os.path.exists(done_path):
    done = {int(x) for x in open(done_path).read().split()}

chunks = [i for i in range((size + CHUNK - 1) // CHUNK) if i not in done]
lock = threading.Lock()
print(f"{len(chunks)} chunks left of {(size + CHUNK - 1) // CHUNK}", flush=True)


def fetch(idx):
    start = idx * CHUNK
    end = min(size, start + CHUNK) - 1
    want = end - start + 1
    for _ in range(20):
        r = subprocess.run(['curl', '-sL', '--fail', '--max-time', '900', '-r', f'{start}-{end}', url],
                           capture_output=True)
        if r.returncode == 0 and len(r.stdout) == want:
            with open(out, 'r+b') as f:
                f.seek(start)
                f.write(r.stdout)
            with lock:
                with open(done_path, 'a') as d:
                    d.write(f'{idx}\n')
                done.add(idx)
                print(f"chunk {idx} ok ({len(done)} done)", flush=True)
            return
    raise RuntimeError(f'chunk {idx} failed')


def worker():
    while True:
        with lock:
            if not chunks:
                return
            idx = chunks.pop(0)
        fetch(idx)


threads = [threading.Thread(target=worker) for _ in range(workers)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("complete", flush=True)
