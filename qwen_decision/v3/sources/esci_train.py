"""Amazon Shopping Queries (ESCI) train split as a 4-way Exact / Substitute / Complement / Irrelevant choice.

Upstream: amazon-science/esci-data `shopping_queries_dataset/` (Apache-2.0) at the commit the index
pins (its git-LFS parquet files are already in the suite's raw copy). Rows with split == "train"
only: the index scores 5,000 pairs from split == "test" (638,016 test rows), and ESCI's train and
test queries are disjoint by construction. Products can appear in both splits under different
queries, so the contamination filter may drop pairs whose product text matches an index row.

Sampling (cap 15,000): stratified by label with square-root-proportional shares (natural train
shares E 65% / S 22% / C 3% / I 10% become roughly 46 / 27 / 10 / 18), locales (us / es / jp)
proportional within each label, at most 2 products per query. Long product fields are truncated,
and a quarter of the states show only title, brand and colour.

Brand and colour are folded into one "details" line: as separate fields they are 1-5 word strings
that coincide with index product rows (a trial run lost ~620 pairs to brand/colour matches alone,
none of them a test query). Title, bullets and description stay separate fields, so a product
shared with an index pair is still caught by the filter.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import random

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from qwen_decision.v3.sources import common

NAME = "esci-train"
REPO = common.ROOT / "external/decision-index/work/artifacts/benchmark-suite/raw/repos/esci"
COMMIT = "7916cdf6ab75a462e77f20ab40428a10923998d5"
CAP = 15_000
PER_QUERY = 2
FIELD_CHARS = 700
LABELS = {
    "E": ("Exact match", "The product is what the shopper searched for and meets every stated requirement."),
    "S": ("Substitute", "Not exactly what was asked for (a detail such as brand, size or a feature differs), but it could be bought instead."),
    "C": ("Complement", "Not the requested item, but something that goes with it, such as an accessory or add-on."),
    "I": ("Irrelevant", "Does not serve the search need at all, or misses a central requirement of the query."),
}
PHRASINGS = [
    "How well does this product match what the shopper searched for?",
    "A customer typed this search query and the store returned the product below. Grade the match.",
    "Judge this search result: is the product an exact match, a usable substitute, a complementary item, or irrelevant to the query?",
    "Rate the relevance of the listed product to the shopping query.",
    "Would this product satisfy the search? Pick the relevance grade that describes it.",
]
QUERY_KEYS = ["search_query", "query", "shopper_query"]


def clip(text: str | None) -> str | None:
    if not text:
        return None
    text = " ".join(text.split())
    return text if len(text) <= FIELD_CHARS else text[:FIELD_CHARS].rsplit(" ", 1)[0] + " …"


def sample(rng: random.Random) -> list[dict]:
    examples = pq.read_table(REPO / "shopping_queries_dataset/shopping_queries_dataset_examples.parquet",
                             columns=["example_id", "query", "query_id", "product_id", "product_locale", "esci_label", "split"],
                             filters=[("split", "=", "train")]).to_pylist()
    counts = Counter(r["esci_label"] for r in examples)
    root = {k: math.sqrt(v) for k, v in counts.items()}
    quota = {k: round(CAP * v / sum(root.values())) for k, v in root.items()}
    rng.shuffle(examples)
    per_query, taken, out = defaultdict(int), Counter(), []
    for r in examples:  # a shuffled pass keeps locale shares proportional within each label
        if taken[r["esci_label"]] >= quota[r["esci_label"]] or per_query[r["query_id"]] >= PER_QUERY:
            continue
        per_query[r["query_id"]] += 1
        taken[r["esci_label"]] += 1
        out.append(r)
    return out


def build(meta: dict):
    rng = random.Random(37)
    pairs = sample(rng)
    meta["label_counts"] = dict(Counter(r["esci_label"] for r in pairs))
    meta["locale_counts"] = dict(Counter(r["product_locale"] for r in pairs))
    wanted = {r["product_id"] for r in pairs}
    products = pq.read_table(REPO / "shopping_queries_dataset/shopping_queries_dataset_products.parquet")
    products = products.filter(pc.is_in(products["product_id"], value_set=pa.array(sorted(wanted))))
    catalog = {(p["product_locale"], p["product_id"]): p for p in products.to_pylist()}
    for r in pairs:
        p = catalog[(r["product_locale"], r["product_id"])]
        fields = ["product_title"] + (["product_bullet_point", "product_description"] if rng.random() < .75 else [])
        product = {k.removeprefix("product_"): clip(p[k]) for k in fields if clip(p.get(k))}
        if "title" not in product:
            continue
        details = [f"{name}: {clip(p[k])}" for k, name in (("product_brand", "brand"), ("product_color", "colour")) if clip(p.get(k))]
        if details:
            product["details"] = "; ".join(details)
        state = {rng.choice(QUERY_KEYS): r["query"], "product": product}
        keys = list(LABELS)
        if rng.random() < .5:
            rng.shuffle(keys)
        options = {LABELS[k][0]: LABELS[k][1] for k in keys} if rng.random() < .7 else [LABELS[k][0] for k in keys]
        question, target = common.choice(state, common.instruction(rng, PHRASINGS), options, keys.index(r["esci_label"]))
        yield common.example(NAME, f"train:{r['example_id']}", "relevance", state, question, target, "train")


if __name__ == "__main__":
    meta = {"upstream": "amazon-science/esci-data shopping_queries_dataset", "revision": COMMIT, "split": "train",
            "licence": "Apache-2.0", "cap": CAP, "sampling": "label-stratified, sqrt-proportional label shares, <=2 products per query"}
    rows = list(build(meta))
    common.write(NAME, rows, meta)
