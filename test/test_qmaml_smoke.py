import unittest

from qmaml_hep.data import create_meta_tasks
from qmaml_hep.model import QuantumClassifier
from qmaml_hep.qmaml import evaluate_few_shot, train_qmaml


class TestQmamlSmoke(unittest.TestCase):
    def test_smoke_training_and_eval(self):
        train_tasks, test_tasks = create_meta_tasks(
            n_tasks=4,
            n_support=8,
            n_query=16,
            n_features=4,
            seed=3,
        )

        model = QuantumClassifier(n_qubits=4, n_layers=1)
        logs = train_qmaml(
            model=model,
            tasks=train_tasks,
            steps=2,
            meta_batch_size=2,
            inner_lr=0.1,
            inner_steps=1,
            outer_lr=0.02,
            seed=3,
        )

        metrics = evaluate_few_shot(model, test_tasks, inner_lr=0.1, inner_steps=1)

        self.assertEqual(len(logs), 2)
        self.assertIn("avg_query_accuracy", metrics)
        self.assertGreaterEqual(metrics["avg_query_accuracy"], 0.0)
        self.assertLessEqual(metrics["avg_query_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
