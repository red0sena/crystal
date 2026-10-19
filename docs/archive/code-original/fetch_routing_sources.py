"""Fetch BANKING77 CSVs and a small, diverse Natural Instructions task subset."""
from collections import Counter
import json
import re

from fetch_text_sources import OUT, download, get, write_json

PATTERNS = {
    "Summarization": "summarization",
    "Question Answering": "question_answering",
    "Question Generation": "question_generation",
    "Information Extraction": "extraction",
    "Sentiment Analysis": "sentiment",
    "Textual Entailment": "entailment",
    "Paraphrasing": "paraphras",
    "Text Categorization": "categor",
}

def main():
    repo = "PolyAI-LDN/task-specific-datasets"
    revision = json.loads(get(f"https://api.github.com/repos/{repo}/commits/master"))["sha"]
    files = [download(f"https://raw.githubusercontent.com/{repo}/{revision}/banking_data/{split}.csv", OUT / "banking77-original" / f"{split}.csv") for split in ("train", "test")]
    write_json(OUT / "banking77-original/manifest.json", {"repo": repo, "revision": revision, "files": files})

    repo = "allenai/natural-instructions"
    revision = json.loads(get(f"https://api.github.com/repos/{repo}/commits/master"))["sha"]
    tree = json.loads(get(f"https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1"))
    write_json(OUT / "natural-instructions/tree.json", tree)
    candidates = [x for x in tree["tree"] if x["path"].startswith("tasks/") and x["path"].endswith(".json") and x["size"] < 5_000_000]
    selected = []
    for category, pattern in PATTERNS.items():
        sources = set()
        options = sorted((x for x in candidates if pattern in x["path"]), key=lambda x: (x["size"], x["path"]))
        for entry in options:
            url = f"https://raw.githubusercontent.com/{repo}/{revision}/{entry['path']}"
            # Inspect only a short header before downloading a task's instances.
            import urllib.request
            cached = OUT / "natural-instructions" / entry["path"]
            if cached.exists():
                header = cached.read_bytes()[:32000].decode(errors="ignore")
            else:
                with urllib.request.urlopen(url, timeout=60) as response:
                    # A byte-limited preview may end inside a UTF-8 character.
                    header = response.read(32000).decode(errors="ignore")
            match = re.search(r'"Categories"\s*:\s*(\[[^\]]*\])', header)
            if not match or json.loads(match[1]) != [category]:
                continue
            language = re.search(r'"Input_language"\s*:\s*(\[[^\]]*\])', header)
            if not language or json.loads(language[1]) != ["English"]:
                continue
            source_match = re.search(r'"Source"\s*:\s*(\[[^\]]*\])', header)
            source = tuple(json.loads(source_match[1])) if source_match else (entry["path"],)
            if source in sources:
                continue
            file = download(url, OUT / "natural-instructions" / entry["path"])
            data = json.loads((OUT / "natural-instructions" / entry["path"]).read_text())
            if not data.get("Instances"):
                continue
            selected.append({"category": category, "source": list(source), **file})
            sources.add(source)
            print(f"Routing source: {category}: {entry['path']}", flush=True)
            if len(sources) == 4:
                break
        if len(sources) < 2:
            raise ValueError(f"Not enough source families for {category}: {len(sources)}")
    write_json(OUT / "natural-instructions/manifest.json", {"repo": repo, "revision": revision, "files": selected, "counts": dict(Counter(x["category"] for x in selected)), "selection": "one task per distinct Source within category; English inputs; single category; files <5 MB; up to 4 source families per category"})
    print("Routing sources complete", flush=True)

if __name__ == "__main__":
    main()
