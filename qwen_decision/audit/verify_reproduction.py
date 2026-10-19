"""Offline verification of preserved evidence, unchanged code behavior, and data.

The default verifies all archived SHA256 hashes and compares the executable AST
of the annotated original modules to their exact old source snapshots. --data
additionally checks regenerated dataset bytes against the original split hashes.
No model loading, GPU, network, or third-party Python packages are needed.

The snapshots predate the move from flat root files into qwen_decision/<group>/
and tests/. Before comparing, Relocation undoes exactly those mechanical edits
(package-qualified imports, ROOT depth, code_path lookups, ``-m`` launches);
any other difference still fails the check.

Run from the project root: python3 -m qwen_decision.audit.verify_reproduction
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

from qwen_decision.paths import code_path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PREFIXES = ("qwen_decision.", "tests.")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flat_name(dotted: str) -> str:
    return dotted.rsplit(".", 1)[1] if dotted.startswith(PACKAGE_PREFIXES) else dotted


def root_file(name: str) -> ast.expr:
    """AST of the flat-layout expression ``ROOT / name``."""
    return ast.BinOp(left=ast.Name(id="ROOT", ctx=ast.Load()), op=ast.Div(),
                     right=ast.Constant(value=name, kind=None))


class Relocation(ast.NodeTransformer):
    """Map a relocated module's AST back to its flat root-layout form."""

    def visit_Module(self, node: ast.Module) -> ast.Module:
        node.body = [s for s in node.body
                     if not (isinstance(s, ast.ImportFrom) and s.module == "qwen_decision.paths")]
        self.generic_visit(node)
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        if node.module:
            node.module = flat_name(node.module)
        return node

    def visit_Import(self, node: ast.Import) -> ast.Import:
        for alias in node.names:
            alias.name = flat_name(alias.name)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        # Path(__file__).resolve().parents[k]  ->  Path(__file__).resolve().parent
        self.generic_visit(node)
        value = node.value
        if (isinstance(value, ast.Attribute) and value.attr == "parents"
                and ast.unparse(value.value) == "Path(__file__).resolve()"
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int)):
            return ast.Attribute(value=value.value, attr="parent", ctx=ast.Load())
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # code_path(x)  ->  ROOT / x
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "code_path" and len(node.args) == 1 \
                and not node.keywords:
            return ast.BinOp(left=ast.Name(id="ROOT", ctx=ast.Load()), op=ast.Div(), right=node.args[0])
        return node

    def visit_List(self, node: ast.List) -> ast.List:
        # "-m", "qwen_decision.<group>.<module>"  ->  str(ROOT / "<module>.py")
        self.generic_visit(node)
        elts, i = [], 0
        while i < len(node.elts):
            first, second = node.elts[i], node.elts[i + 1] if i + 1 < len(node.elts) else None
            if (isinstance(first, ast.Constant) and first.value == "-m" and isinstance(second, ast.Constant)
                    and isinstance(second.value, str) and second.value.startswith(PACKAGE_PREFIXES)):
                elts.append(ast.Call(func=ast.Name(id="str", ctx=ast.Load()),
                                     args=[root_file(flat_name(second.value) + ".py")], keywords=[]))
                i += 2
            else:
                elts.append(first)
                i += 1
        node.elts = elts
        return node


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", action="store_true")
    args = parser.parse_args()
    archive = json.loads((ROOT / "docs/archive_manifest.json").read_text())
    for entry in archive["files"]:
        path = ROOT / entry["archive_path"]
        assert path.stat().st_size == entry["bytes"], path
        assert sha(path) == entry["sha256"], path
    snapshots = ROOT / "docs/archive/code-original"
    originals = sorted(snapshots.glob("*.py"))
    for original in originals:
        current = code_path(original.name)
        assert ast.dump(ast.parse(original.read_text()), include_attributes=False) == ast.dump(
            Relocation().visit(ast.parse(current.read_text())), include_attributes=False), current
    print(f"{len(archive['files'])} archived files verified; {len(originals)} original executable Python ASTs "
          "unchanged apart from the package relocation.")
    if args.data:
        checked = 0
        for version in ("text-decisions-v1", "text-decisions-v2"):
            old = json.loads((ROOT / f"docs/archive/data/{version}/manifest.json").read_text())
            for split, spec in old["stats"].items():
                path = ROOT / f"data/{version}/{split}.jsonl"
                assert sha(path) == spec["file_sha256"], path
                with path.open() as handle:
                    assert sum(1 for line in handle if line.strip()) == spec["total"], path
                checked += 1
        old = json.loads((ROOT / "docs/archive/data/unseen-text-v1/manifest.json").read_text())
        for entry in old["datasets"]:
            path = ROOT / f"data/unseen-text-v1/{entry['name']}.jsonl"
            assert sha(path) == entry["sha256"], path
            checked += 1
        old = json.loads((ROOT / "docs/archive/data/external-classification-eval-v1/manifest.json").read_text())
        for spec in old["datasets"]:
            path = ROOT / "data/external-classification-eval-v1" / Path(spec["file"]).name
            assert sha(path) == spec["sha256"], path
            checked += 1
        print(f"{checked} dataset split files match the original bytes exactly.")


if __name__ == "__main__":
    main()
