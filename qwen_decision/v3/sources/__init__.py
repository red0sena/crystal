"""Phase-2 training sources: one module per dataset family, all writing through common.py.

Each builder downloads its pinned source, casts every item as a System One question with a target
distribution, drops anything the contamination filter flags (Decision Index rows, every ToolRet
query, JEVal) and writes data/phase2/<name>.jsonl plus <name>.manifest.json.

  .venv-di/bin/python -m qwen_decision.v3.sources.<module>
"""
