import uuid
import unittest
from pathlib import Path
from src.sst2 import Example, load_tsv, prepare


class SST2Tests(unittest.TestCase):
    def setUp(self):
        self.train = [Example(f"train:{i}", f"Sentence {i}", i % 2) for i in range(200)]
        self.dev = [Example(f"validation:{i}", f"Official dev {i}", i % 2) for i in range(20)]

    def test_policies_and_no_id_leakage(self):
        expected = {"cap_then_holdout": (90, 10, 20), "holdout_then_cap": (100, 20, 20), "capped_70_15_15": (70, 15, 15)}
        for policy, sizes in expected.items():
            splits, manifest = prepare(self.train, self.dev, policy=policy, seed=0, cap=100)
            self.assertEqual(tuple(map(len, splits.values())), sizes)
            ids = [r.source_id for rows in splits.values() for r in rows]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertFalse(manifest["exact_author_split"])
            if policy != "capped_70_15_15":
                self.assertEqual(splits["test"], self.dev)

    def test_reproducibility_and_seed_effect(self):
        args = dict(policy="cap_then_holdout", cap=100)
        a = prepare(self.train, self.dev, seed=0, **args)
        self.assertEqual(a, prepare(self.train, self.dev, seed=0, **args))
        self.assertNotEqual(a, prepare(self.train, self.dev, seed=1, **args))
        self.assertEqual(sum(r.label for r in a[0]["train"]), 45)

    def test_tsv_preserves_text_and_rejects_unlabeled(self):
        p = Path(__file__).parent / ("fixture-" + uuid.uuid4().hex + ".tsv")
        try:
            p.write_text("sentence\tlabel\n  Great Movie!  \t1\nBad.\t0\n", encoding="utf-8")
            self.assertEqual(load_tsv(p, "train")[0].text, "  Great Movie!  ")
            p.write_text("index\tsentence\n0\ttext\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_tsv(p, "train")
        finally:
            p.unlink(missing_ok=True)

    def test_invalid_choices(self):
        with self.assertRaises(ValueError):
            prepare(self.train, self.dev, policy="automatic", seed=0)
        with self.assertRaises(ValueError):
            prepare(self.train, self.dev, policy="cap_then_holdout", seed=0, cap=1000)


if __name__ == "__main__":
    unittest.main()
