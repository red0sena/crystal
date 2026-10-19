"""NLI4CT (SemEval-2024 Task 2) train + dev (1,700 + 200 statements) as entailment / contradiction over trial sections.

Upstream: ai-systems/Task-2-SemEval-2024, `training_data.zip` (train.json, dev.json and the
clinical-trial reports under `CT json/`) at the commit the index pins (the suite rebuild already
cloned it; read-only). Licence: no licence file; SemEval-2024 shared-task data, CTRs from
ClinicalTrials.gov (public). The index scores `test.json` + `gold_test.json` (catalog 42); the
practice test is a test-phase file and is not used. No train/dev statement is a test statement
(measured in the plan), but ~200 train statements are about a trial whose section a test row also
shows; the filter drops a row whose trial text is shared with a test row (reported in the manifest).

The state is the indicated section of the primary trial (and of the secondary trial for comparison
statements), as the index does; the statement goes in the state or the instruction. 60% are a
2-way choice, 40% yes/no with both polarities (is it supported / is it contradicted).
"""
from __future__ import annotations

import json
import random
import zipfile

from qwen_decision.v3.sources import common

NAME = "nli4ct-train"
REPO = common.ROOT / "external/decision-index/work/artifacts/benchmark-suite/raw/repos/nli4ct"
COMMIT = "7f32fa6c7db43e577f22e9fc6c28eef1cc6d223e"
# Option sets in (entailment, contradiction) order.
LABELS = [
    [("entailment", "the trial report backs the statement"), ("contradiction", "the trial report refutes the statement")],
    [("true", None), ("false", None)],
    [("supported", "the evidence confirms it"), ("refuted", "the evidence shows it is wrong")],
    [("consistent with the trial data", None), ("inconsistent with the trial data", None)],
]
CHOICE = [
    "Is the statement below borne out by the clinical trial section(s) shown, or contradicted by them?\n{s}",
    "Check this claim against the trial report: {s}",
    "Using only the trial information given, classify the statement.\n{s}",
    "Does the clinical trial evidence confirm or refute the following? {s}",
]
CHOICE_GENERIC = [
    "Judge the statement against the clinical trial data that accompanies it.",
    "Decide whether the trial record supports or refutes the statement.",
]
ENTAILED = [
    "Is this statement correct according to the clinical trial report(s)?\n{s}",
    "The trial data shown supports the following claim: {s}",
    "Based on the trial section(s), is the following true? {s}",
]
CONTRADICTED = [
    "Does the trial evidence contradict this statement?\n{s}",
    "The following claim is refuted by the clinical trial record: {s}",
    "According to the trial section(s), is this statement false? {s}",
]


def section(trial: dict, name: str) -> str:
    return "\n".join(line.rstrip() for line in trial[name])


def build():
    rng = random.Random(2026101642)
    with zipfile.ZipFile(REPO / "training_data.zip") as z:
        trials = {}

        def trial(nct):
            if nct not in trials:
                trials[nct] = json.loads(z.read(f"CT json/{nct}.json"))
            return trials[nct]

        for split in ("train", "dev"):
            for uid, r in json.loads(z.read(f"{split}.json")).items():
                name = r["Section_id"]
                parts = [("primary trial", r["Primary_id"])] + ([("secondary trial", r["Secondary_id"])] if "Secondary_id" in r else [])
                texts = [(role, nct, section(trial(nct), name)) for role, nct in parts]
                statement = r["Statement"].strip()
                inline = rng.random() < .65
                if rng.random() < .5:
                    state = {role: {"trial id": nct, "section": name, "text": text} for role, nct, text in texts}
                else:
                    state = "\n\n".join(f"{role.title()} ({nct}), {name}:\n{text}" for role, nct, text in texts)
                if not inline:
                    state = {"trial data": state, "statement": statement} if isinstance(state, str) else {**state, "statement": statement}
                entailed = r["Label"] == "Entailment"
                if rng.random() < .6:
                    labels = rng.choice(LABELS)
                    order = [0, 1] if rng.random() < .5 else [1, 0]
                    criteria = {labels[i][0]: labels[i][1] for i in order}
                    instructions = rng.choice(CHOICE).format(s=statement) if inline else rng.choice(CHOICE_GENERIC)
                    question, target = common.choice(state, instructions, criteria, order.index(0 if entailed else 1))
                elif rng.random() < .5:
                    text = rng.choice(ENTAILED).format(s=statement) if inline else "Is the statement supported by the trial data?"
                    question, target = common.noul(text, entailed)
                else:
                    text = rng.choice(CONTRADICTED).format(s=statement) if inline else "Does the trial data contradict the statement?"
                    question, target = common.noul(text, not entailed)
                yield common.example(NAME, f"{split}:{uid}", "nli", state, question, target, split)


if __name__ == "__main__":
    common.write(NAME, build(), {"upstream": "ai-systems/Task-2-SemEval-2024 (training_data.zip)", "revision": COMMIT,
                                 "split": "train+dev",
                                 "licence": "no licence file (SemEval-2024 Task 2 data; CTRs from ClinicalTrials.gov)"})
