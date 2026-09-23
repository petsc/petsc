#!/usr/bin/env bash
# Level-1 regression suite (guards self-improvement against drift).
#
# Rebuilds and reruns every verified component and re-validates the study/index
# artifacts against their schemas. A component promotion or skill edit is trusted
# only if this still passes. Requires PETSC_DIR/PETSC_ARCH set and mpiexec.
#
# Each component is expected to expose a `make run` target that writes a
# convergence.csv with columns h,L2,Linf (see components/README.md); the last two
# refinement levels must show ~2nd-order (or better) convergence.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== [1/2] Components: build, run, check convergence =="
shopt -s nullglob
components=(components/*/makefile)
if [ ${#components[@]} -eq 0 ]; then
  echo "   (no components to check)"
fi
for mk in "${components[@]}"; do
  comp="$(dirname "$mk")"
  name="$(basename "$comp")"
  echo "-- component $name"
  ( cd "$comp" && make -s run )
  python3 - "$ROOT/$comp/convergence.csv" "$name" <<'PY'
import csv, math, sys
csv_path, name = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(csv_path)))
def order(a, b, key):
    return math.log(float(a[key]) / float(b[key])) / math.log(float(a["h"]) / float(b["h"]))
oL2   = order(rows[-2], rows[-1], "L2")
oLinf = order(rows[-2], rows[-1], "Linf")
print(f"   observed order: L2={oL2:.3f}, Linf={oLinf:.3f} (expected >= ~2.0)")
sys.exit(0 if oL2 >= 2.0 - 0.1 and oLinf >= 2.0 - 0.1 else 1)
PY
  echo "   [ ok ] convergence holds for $name"
done

echo "== [2/2] Validate study + case-index artifacts against schemas =="
python3 tests/validate.py

echo "== REGRESSION PASSED =="
