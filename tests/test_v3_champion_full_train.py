import json
from pathlib import Path
import pickle
import tempfile
import unittest
from collections import Counter

import numpy as np

from qwen_decision.v3.champion_full_inputs import IndexedRows
from qwen_decision.v3.champion_full_train import Schedule


class FullTrainingDataTests(unittest.TestCase):
    def test_every_text_row_is_kept_including_partial_batch_and_resume_order(self):
        texts = [{"id": "t"+str(i), "modality": "text"} for i in range(18)]
        images = [{"id": "i"+str(i), "modality": "image"} for i in range(3)]
        schedule = Schedule(texts, images, epochs=2, seed=8)
        self.assertEqual(len(schedule), 4)
        for epoch in range(2):
            rows = [r for step in range(epoch*2, (epoch+1)*2) for r, _ in schedule[step]]
            self.assertEqual(Counter(r["id"] for r in rows if r["modality"] == "text"), {r["id"]: 1 for r in texts})
            self.assertEqual(sum(r["modality"] == "image" for r in rows), 8)
        self.assertEqual(len(schedule[1]), 6)
        restarted = Schedule(texts, images, epochs=2, seed=8)
        for i in range(1, len(schedule)):
            self.assertEqual(schedule[i], restarted[i])
        image_counts = Counter(r["id"] for i in range(4) for r, _ in schedule[i] if r["modality"] == "image")
        self.assertEqual(sorted(image_counts.values()), [5, 5, 6])

    def test_indexed_parts_and_worker_pickle_preserve_rows_and_gold_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            files, indices = [], []
            for part in range(2):
                path = Path(tmp)/f"{part}.jsonl"
                entries = []
                with path.open("wb") as file:
                    for i in range(2):
                        row = {"id": f"{part}:{i}", "modality": "text", "source": "test", "state": "x",
                               "question": {"type": "choice", "instructions": "Pick", "criteria": {"x": None, "y": None}},
                               "target": [.2, .8], "target_kind": "vote_distribution", "label_origin": "human"}
                        raw = (json.dumps(row)+"\n").encode()
                        entries.append((file.tell(), len(raw), 0))
                        file.write(raw)
                index = path.with_suffix(".npy")
                np.save(index, np.asarray(entries, dtype=np.uint64))
                files.append(path)
                indices.append(index)
            reader = IndexedRows(files, indices)
            self.assertEqual(reader[2]["id"], "1:0")
            self.assertEqual(reader[0]["target"], [0., 1.])
            restored = pickle.loads(pickle.dumps(reader))
            self.assertEqual([restored[i]["id"] for i in range(4)], ["0:0", "0:1", "1:0", "1:1"])
            for handle in list(reader.handles.values())+list(restored.handles.values()):
                handle.close()


if __name__ == "__main__":
    unittest.main()
