#!/usr/bin/env bash
# Experiment pipeline. Finished steps leave a marker, so a rerun resumes where it stopped.
#
# Environment:
#   PY    python interpreter        (default: python)
#   BASE  Qwen3-1.7B weights        (default: models/Qwen3-1.7B)
#   BERT  bert-base-chinese weights (default: models/bert-base-chinese)
#   DATA  output of 01_prepare_data (default: data)
#   RUNS  adapter output directory  (default: runs)
#   LOG   log directory             (default: logs)
#
# Usage: bash scripts/run_all.sh
set -u
cd "$(dirname "$0")/.."

PY="${PY:-python}"; BASE="${BASE:-models/Qwen3-1.7B}"; BERT="${BERT:-models/bert-base-chinese}"
DATA="${DATA:-data}"; RUNS="${RUNS:-runs}"; LOG="${LOG:-logs}"
mkdir -p "$RUNS" "$LOG/.done"

step() {
  local name="$1"; shift
  if [ -f "$LOG/.done/$name" ]; then echo "[skip] $name"; return 0; fi
  echo "[$(date '+%H:%M:%S')] START $name"
  if "$PY" -I "$@" > "$LOG/$name.log" 2>&1; then
    touch "$LOG/.done/$name"
    echo "[$(date '+%H:%M:%S')] DONE  $name"
  else
    echo "[$(date '+%H:%M:%S')] FAIL  $name, see $LOG/$name.log"
  fi
}

evaluate() { step "eval_$1" scripts/02_evaluate_qwen.py --base "$BASE" --data "$DATA" --arm "$1" "${@:2}"; }

# Pretrained baselines, instruction tuning, and the BERT reproduction.
evaluate pretrained_zero
evaluate pretrained_few --shots
step sft_s42 scripts/03_train_sft.py --base "$BASE" --data "$DATA" --out "$RUNS/sft_s42" --run-name sft_s42 --seed 42
evaluate sft_s42 --adapter "$RUNS/sft_s42"
for strategy in head top2 full layerwise; do
  step "bert_${strategy}_detect_s42" scripts/04_train_bert.py --bert "$BERT" --data "$DATA" --task detect \
    --strategy "$strategy" --seed 42 --dump-embeddings "$RUNS/embeddings"
done
step bert_full_type_s42 scripts/04_train_bert.py --bert "$BERT" --data "$DATA" --task type --strategy full --seed 42
step bert_layerwise_type_s42 scripts/04_train_bert.py --bert "$BERT" --data "$DATA" --task type --strategy layerwise --seed 42

# How little data does instruction tuning need?
step sft_frac0.01_s42 scripts/03_train_sft.py --base "$BASE" --data "$DATA" --out "$RUNS/sft_frac0.01_s42" \
  --run-name sft_frac0.01_s42 --fraction 0.01 --epochs 6 --seed 42
evaluate sft_frac0.01_s42 --adapter "$RUNS/sft_frac0.01_s42"
step sft_frac0.05_s42 scripts/03_train_sft.py --base "$BASE" --data "$DATA" --out "$RUNS/sft_frac0.05_s42" \
  --run-name sft_frac0.05_s42 --fraction 0.05 --epochs 4 --dev-limit 200 --seed 42
evaluate sft_frac0.05_s42 --adapter "$RUNS/sft_frac0.05_s42"

# Alignment tuning on top of the instruction-tuned model.
step pairs scripts/05_make_pairs.py --base "$BASE" --adapter "$RUNS/sft_s42" --data "$DATA"
step dpo_beta0.1 scripts/06_train_dpo.py --base "$BASE" --adapter "$RUNS/sft_s42" --data "$DATA" \
  --out "$RUNS/dpo_beta0.1" --run-name dpo_beta0.1 --beta 0.1 --eval-every 100 --dev-limit 300
evaluate dpo_beta0.1 --adapter "$RUNS/dpo_beta0.1"
# Same run at a tenth of the learning rate, to separate a step-size effect from the data effect.
step dpo_beta0.1_lowlr scripts/06_train_dpo.py --base "$BASE" --adapter "$RUNS/sft_s42" --data "$DATA" \
  --out "$RUNS/dpo_lowlr" --run-name dpo_beta0.1_lowlr --beta 0.1 --lr 2e-6 --eval-every 100 --dev-limit 300
for reward in rubric naive; do
  step "ppo_${reward}" scripts/07_train_ppo.py --base "$BASE" --adapter "$RUNS/sft_s42" --data "$DATA" \
    --out "$RUNS/ppo_${reward}" --run-name "ppo_${reward}" --reward "$reward" \
    --updates 40 --eval-every 20 --dev-limit 200
  evaluate "ppo_${reward}" --adapter "$RUNS/ppo_${reward}"
done

echo "PIPELINE FINISHED"
