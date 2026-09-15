"""Download immutable HF assets into an explicit cache directory, never train."""
import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def download_weights(url, dest):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=60) as response:
        if response.status != 206:
            raise RuntimeError("Range download unsupported")
        total = int(response.headers["Content-Range"].split("/")[-1])
    start = dest.stat().st_size if dest.exists() else 0
    if start > total:
        raise RuntimeError("Cached file too large")
    ranges = [(i, min(i + 8_000_000, total) - 1) for i in range(start, total, 8_000_000)]
    def part(bounds):
        lo, hi = bounds
        req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
        with urllib.request.urlopen(req, timeout=120) as response:
            expected = f"bytes {lo}-{hi}/{total}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise RuntimeError("Incorrect range response")
            data = response.read()
        if len(data) != hi - lo + 1:
            raise RuntimeError("Incomplete range")
        return data
    with ThreadPoolExecutor(max_workers=8) as pool, dest.open("ab") as handle:
        for data in pool.map(part, ranges):
            handle.write(data)
            handle.flush()
            print(f"Weights {handle.tell()}/{total} bytes", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    records = {}
    for kind, repo, files in [
        ("models", "roberta-base", ["config.json", "merges.txt", "vocab.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]),
        ("datasets", "stanfordnlp/sst2", ["data/train-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet"]),
    ]:
        info = json.load(urllib.request.urlopen(f"https://huggingface.co/api/{kind}/{repo}?blobs=true"))
        sha = info["sha"]
        records[repo] = {"revision": sha, "files": {}}
        for name in files:
            dest = a.output / ("model" if kind == "models" else "data") / Path(name).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            prefix = "datasets/" if kind == "datasets" else ""
            url = f"https://huggingface.co/{prefix}{repo}/resolve/{sha}/{name}"
            print(f"Downloading {repo}/{name}", flush=True)
            if name == "model.safetensors":
                download_weights(url, dest)
            else:
                urllib.request.urlretrieve(url, dest)
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()
            expected = next(x for x in info["siblings"] if x["rfilename"] == name).get("lfs", {}).get("sha256")
            if expected and digest != expected:
                raise RuntimeError(f"Asset checksum mismatch: {name}")
            records[repo]["files"][name] = {"url": url, "sha256": digest, "upstream_lfs_sha256": expected}
    (a.output / "provenance.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
