#!/usr/bin/env bash
# Run stages explicitly: rebuilding data is cheap; training/evaluation uses GPU.
# This script never deletes runs. Historical records live under docs/archive.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"
TASK_PY="$TASK_ROOT/.venv/bin/python"

stage_env() {
    uv sync --frozen
}
stage_sources() {
    python3 -m qwen_decision.sources.restore_sources
}
stage_data() {
    "$TASK_PY" -m qwen_decision.prepare.prepare_text_decisions
    "$TASK_PY" -m qwen_decision.audit.audit_text_decisions
    "$TASK_PY" -m qwen_decision.prepare.prepare_text_decisions_v2
    "$TASK_PY" -m qwen_decision.audit.audit_text_decisions_v2
    "$TASK_PY" -m qwen_decision.prepare.prepare_unseen_text_evaluation
    "$TASK_PY" -m qwen_decision.prepare.prepare_external_classification_evaluation
    python3 -m qwen_decision.audit.verify_reproduction --data
}
stage_train_v1() {
    # v1 originally used the PyTorch reference DeltaNet path. uv sync removes
    # optional v2 kernel packages if a previous stage installed them.
    uv sync --frozen
    "$TASK_PY" -m qwen_decision.train.train_text_decisions --preflight --output training/reproduction-v1-preflight
    "$TASK_PY" -m qwen_decision.train.run_text_training_pipeline --output training/text-decision-full-v1
}
stage_train_v2() {
    uv pip install --python "$TASK_PY" flash-linear-attention==0.5.2 fla-core==0.5.2
    "$TASK_PY" -m qwen_decision.train.train_text_decisions_v2 --preflight --output training/reproduction-v2-preflight
    "$TASK_PY" -m qwen_decision.train.train_text_decisions_v2 --output training/text-decision-v2
}
stage_evaluate_v2() {
    uv pip install --python "$TASK_PY" flash-linear-attention==0.5.2 fla-core==0.5.2
    for TASK_MODEL_NAME in original v1 v2; do
        case "$TASK_MODEL_NAME" in
            original) TASK_MODEL_PATH="$TASK_ROOT/model" ;;
            v1) TASK_MODEL_PATH="$TASK_ROOT/training/text-decision-full-v1/model" ;;
            v2) TASK_MODEL_PATH="$TASK_ROOT/training/text-decision-v2/model" ;;
        esac
        "$TASK_PY" -m qwen_decision.evaluate.evaluate_v2 --model "$TASK_MODEL_PATH" --output "results/v2-eval/$TASK_MODEL_NAME"
        "$TASK_PY" -m qwen_decision.analysis.calibrate_v2 --predictions "results/v2-eval/$TASK_MODEL_NAME/calibration.jsonl" \
            --output "results/v2-eval/$TASK_MODEL_NAME/calibration-fit.json"
    done
    cp results/v2-eval/v2/calibration-fit.json training/text-decision-v2/model/calibration.json
    "$TASK_PY" -m qwen_decision.audit.audit_text_decisions_v2
    "$TASK_PY" -m qwen_decision.analysis.report_v2
    "$TASK_PY" -m tests.test_scorer_v2 --model training/text-decision-v2/model --output results/v2-eval/shared-state-check.json
}

case "${1:---plan}" in
    --plan)
        printf '%s\n' \
          'env: uv sync --frozen (Python packages only)' \
          'sources: restore pinned model/raw data, verify SHA256 (~2.15 GiB)' \
          'data: rebuild v1/v2/unseen/external JSONL and run audits (CPU)' \
          'train-v1: reference kernels; two independent full-language runs; dev selection; 16-task sample evaluation (GPU)' \
          'train-v2: install FLA 0.5.2; two independent full-language runs; dev selection (GPU)' \
          'evaluate-v2: same v2 protocol for original/v1/v2; fit held-out dev temperatures; report; shared-state check (GPU)' \
          'all: execute the six stages in this order; may take many hours and recreate tens of GiB'
        ;;
    env) stage_env ;;
    sources) stage_sources ;;
    data) stage_data ;;
    train-v1) stage_train_v1 ;;
    train-v2) stage_train_v2 ;;
    evaluate-v2) stage_evaluate_v2 ;;
    all) stage_env; stage_sources; stage_data; stage_train_v1; stage_train_v2; stage_evaluate_v2 ;;
    *) printf '%s\n' 'Usage: bash reproduce.sh {--plan|env|sources|data|train-v1|train-v2|evaluate-v2|all}' >&2; exit 2 ;;
esac
