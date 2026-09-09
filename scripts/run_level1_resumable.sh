#!/usr/bin/env bash
# Run the level 1 sweep until every task has a result, restarting after crashes.
#
# The benchmark dies part-way with a segmentation fault often enough to matter:
# level1_final_19 stopped at 15 of 53 and level1_final_22 at 12, while
# level1_final_20 and _21 finished. It lands at the boundary where FAISS has just
# serialised an index and a torch model is loaded next, and it did not reproduce
# when the same task was re-run alone, so it is intermittent rather than tied to
# any task.
#
# A segfault leaves no Python traceback and no error field, and the runner has no
# resume, so each crash costs the whole run. This wrapper turns that into the
# cost of one task: completed tasks are read back from their per-task JSON and
# the remaining ids are passed to the next attempt.
#
# PYTHONFAULTHANDLER is on so the next crash prints the Python stack it died in,
# which is the one thing missing from every crash so far.
#
# Usage:  bash scripts/run_level1_resumable.sh [log_name] [max_attempts]

set -u

LOG_NAME="${1:-level1_final_22}"
MAX_ATTEMPTS="${2:-12}"
ROOT="/c/SCP"
PY="${ROOT}/venv312/Scripts/python.exe"
TASK_DIR="${ROOT}/outputs/${LOG_NAME}/tasks"
RUN_LOG="${ROOT}/outputs/${LOG_NAME}.log"
TOTAL=53

export PYTHONFAULTHANDLER=1
export PYTHONIOENCODING=utf-8
# Both faiss-cpu and torch bring their own OpenMP runtime and each reports 20
# threads. Two runtimes in one process is a known source of crashes at exactly
# this boundary. Capping faiss is the cheap half of the mitigation; it is not
# proven to be the cause, because the crash has never reproduced on demand.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

completed_ids() {
    [ -d "${TASK_DIR}" ] || return 0
    "${PY}" - "${TASK_DIR}" <<'PYEOF'
import glob, json, os, sys
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    try:
        print(json.loads(open(path, encoding="utf-8").read()).get("task_id") or "")
    except Exception:
        pass
PYEOF
}

remaining_ids() {
    "${PY}" - "$@" <<'PYEOF'
import glob, sys
import pandas as pd
done = {line.strip() for line in sys.argv[1:] if line.strip()}
frame = pd.read_parquet(glob.glob("c:/SCP/data/gaia/2023/validation/metadata.level1.parquet")[0])
todo = [str(row["task_id"]) for _, row in frame.iterrows() if str(row["task_id"]) not in done]
print(",".join(todo))
PYEOF
}

for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    mapfile -t DONE < <(completed_ids)
    COUNT="${#DONE[@]}"
    if [ "${COUNT}" -ge "${TOTAL}" ]; then
        echo "[resume] all ${TOTAL} tasks present"
        break
    fi
    TODO="$(remaining_ids "${DONE[@]:-}")"
    if [ -z "${TODO}" ]; then
        echo "[resume] nothing left to run"
        break
    fi
    echo "[resume] attempt ${attempt}: ${COUNT}/${TOTAL} done, running the rest"
    "${PY}" -m benchmark.gaia.gaia_runner \
        --level 1 \
        --task-ids "${TODO}" \
        --stage1-runs-per-agent 3 \
        --evidence-prepare true \
        --enable-evidence-driven-search \
        --enable-agent-tool-use \
        --bypass-search-labeler \
        --agent-prepared-search-budget 2 \
        --log-name "${LOG_NAME}" >> "${RUN_LOG}" 2>&1
    STATUS=$?
    echo "[resume] attempt ${attempt} exited ${STATUS}"
    # 139 is SIGSEGV; anything non-zero gets another attempt while tasks remain.
    if [ "${STATUS}" -eq 0 ]; then
        mapfile -t DONE < <(completed_ids)
        [ "${#DONE[@]}" -ge "${TOTAL}" ] && break
        echo "[resume] exited cleanly with ${#DONE[@]}/${TOTAL} — continuing"
    fi
done

mapfile -t DONE < <(completed_ids)
echo "[resume] finished with ${#DONE[@]}/${TOTAL} tasks"
