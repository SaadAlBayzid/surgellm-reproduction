import unittest
import torch
from transformers import RobertaConfig, RobertaForSequenceClassification
from src.roberta_baseline import classification_metrics, learning_rate_factor, evaluate


class BaselineTests(unittest.TestCase):
    def test_metrics_fixed_two_classes(self):
        self.assertEqual(classification_metrics([0, 1], [0, 1]), {"accuracy": 1.0, "macro_f1": 1.0})
        self.assertAlmostEqual(classification_metrics([0, 1], [0, 0])["macro_f1"], 1 / 3)
        self.assertEqual(classification_metrics([0], [0])["macro_f1"], .5)

    def test_schedule_boundaries(self):
        self.assertEqual(learning_rate_factor(0, 10, 2), 0)
        self.assertEqual(learning_rate_factor(2, 10, 2), 1)
        self.assertEqual(learning_rate_factor(10, 10, 2), 0)

    def test_eval_example_weighting_and_no_grad(self):
        # Tiny random model is only a unit fixture; real roberta-base smoke is separate.
        torch.manual_seed(0)
        model = RobertaForSequenceClassification(RobertaConfig(vocab_size=20, hidden_size=8,
                 num_hidden_layers=1, num_attention_heads=2, intermediate_size=16, num_labels=2))
        batch = {"input_ids": torch.tensor([[0, 5, 2], [0, 6, 2], [0, 7, 2]]),
                 "attention_mask": torch.ones(3, 3, dtype=torch.long), "labels": torch.tensor([0, 1, 0])}
        full = evaluate(model, [batch], "cpu")
        partial = evaluate(model, [{k: v[:2] for k, v in batch.items()}, {k: v[2:] for k, v in batch.items()}], "cpu")
        self.assertAlmostEqual(full["loss"], partial["loss"], places=6)
        self.assertEqual(full["macro_f1"], partial["macro_f1"])
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
