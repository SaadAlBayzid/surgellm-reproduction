"""Convert the downloaded canonical SST-2 parquet fields to the validated TSV loader format."""
import argparse
import csv
from pathlib import Path
import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    a = p.parse_args()
    for split, dest in [("train", "train.tsv"), ("validation", "dev.tsv")]:
        rows = pq.read_table(a.assets / "data" / f"{split}-00000-of-00001.parquet").to_pylist()
        with (a.assets / dest).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sentence", "label"], delimiter="\t")
            writer.writeheader()
            writer.writerows({"sentence": row["sentence"], "label": row["label"]} for row in rows)
        print(split, len(rows))


if __name__ == "__main__":
    main()
