#!/usr/bin/env python3
"""Dry-run validator: check each pipeline artifact against its contract schema,
verify cross-references, and independently recompute convergence orders.

Discovers every worked study under examples/ (any directory holding a
problem-spec.json, or — for the programming lane, which emits no problem spec —
a results-manifest.json) and always validates components/case-index.json, so it
stays correct as studies are added or removed."""
import json, sys, math, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONTRACTS = ROOT / "contracts"
EXAMPLES = ROOT / "examples"

PAIRS = [
    ("problem-spec.json",        "problem-spec.schema.json"),
    ("numerical-plan.json",      "numerical-plan.schema.json"),
    ("vis-spec.json",            "vis-spec.schema.json"),
    ("results-manifest.json",    "results-manifest.schema.json"),
    ("numerical-assessment.json","numerical-assessment.schema.json"),
    ("analysis-report.json",     "analysis-report.schema.json"),
]

def load(p): return json.loads(pathlib.Path(p).read_text())

def jtype_ok(v, t):
    if t == "object":  return isinstance(v, dict)
    if t == "array":   return isinstance(v, list)
    if t == "string":  return isinstance(v, str)
    if t == "boolean": return isinstance(v, bool)
    if t == "integer": return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":  return isinstance(v, (int, float)) and not isinstance(v, bool)
    return True

def validate(inst, schema, path, errs):
    """Recursive Draft-2020-12 subset validator."""
    t = schema.get("type")
    if t and not jtype_ok(inst, t):
        errs.append(f"{path or '<root>'}: expected type {t}, got {type(inst).__name__}")
        return
    if "const" in schema and inst != schema["const"]:
        errs.append(f"{path}: expected const {schema['const']!r}, got {inst!r}")
    if "enum" in schema and inst not in schema["enum"]:
        errs.append(f"{path}: {inst!r} not in enum {schema['enum']}")
    if isinstance(inst, dict):
        for req in schema.get("required", []):
            if req not in inst:
                errs.append(f"{path}: missing required '{req}'")
        props = schema.get("properties", {})
        addl = schema.get("additionalProperties", True)
        for k, v in inst.items():
            kp = f"{path}.{k}" if path else k
            if k in props:
                validate(v, props[k], kp, errs)
            elif addl is False:
                errs.append(f"{kp}: additional property not allowed")
            elif isinstance(addl, dict):
                validate(v, addl, kp, errs)
    if isinstance(inst, list):
        if "minItems" in schema and len(inst) < schema["minItems"]:
            errs.append(f"{path}: needs >= {schema['minItems']} items, got {len(inst)}")
        if "items" in schema:
            for i, el in enumerate(inst):
                validate(el, schema["items"], f"{path}[{i}]", errs)

def ref(src, src_key, dst, dst_key, label):
    """Return a (label, ok) cross-reference check, or None when either endpoint
    is absent — the referring key or the referent id. A study that omits an
    optional artifact is skipped here rather than crashing the run."""
    if src is not None and dst is not None and src_key in src and dst_key in dst:
        return (label, src[src_key] == dst[dst_key])
    return None

def check_study(study):
    """Validate one worked study; return True if everything passed.

    Only problem-spec.json and results-manifest.json are guaranteed. The other
    artifacts are optional by design: visualization is on-demand (D20), and the
    programming lane skips numerical analysis and MMS (D21). Absent artifacts are
    skipped; each cross-check runs only when both sides are present."""
    ok = True
    print(f"\n### study: {study.name}")
    print("--- schema validation ---")
    loaded = {}
    for artifact, schema in PAIRS:
        ap = study / artifact
        if not ap.is_file():
            print(f"[skip] {artifact} (not present)")
            continue
        a = load(ap)
        loaded[artifact] = a
        s = load(CONTRACTS / schema)
        errs = []
        validate(a, s, "", errs)
        if errs:
            ok = False
            print(f"[FAIL] {artifact}")
            for e in errs[:12]:
                print(f"       {e}")
        else:
            print(f"[ ok ] {artifact}")

    print("--- cross-reference integrity ---")
    ps  = loaded.get("problem-spec.json")
    np_ = loaded.get("numerical-plan.json")
    vs  = loaded.get("vis-spec.json")
    rm  = loaded.get("results-manifest.json")
    na  = loaded.get("numerical-assessment.json")
    ar  = loaded.get("analysis-report.json")
    checks = [c for c in (
        ref(np_, "problem_spec_id",     ps,  "id", "numerical-plan.problem_spec_id -> problem-spec.id"),
        ref(vs,  "problem_spec_id",     ps,  "id", "vis-spec.problem_spec_id -> problem-spec.id"),
        ref(vs,  "numerical_plan_id",   np_, "id", "vis-spec.numerical_plan_id -> numerical-plan.id"),
        ref(rm,  "numerical_plan_id",   np_, "id", "results.numerical_plan_id -> numerical-plan.id"),
        ref(rm,  "vis_spec_id",         vs,  "id", "results.vis_spec_id -> vis-spec.id"),
        ref(na,  "results_manifest_id", rm,  "id", "assessment.results_manifest_id -> results.id"),
        ref(ar,  "results_manifest_id", rm,  "id", "analysis.results_manifest_id -> results.id"),
    ) if c is not None]
    if not checks:
        print("  (no cross-references to check)")
    for name, good in checks:
        print(f"[{'ok' if good else 'FAIL'}] {name}")
        ok = ok and good

    print("--- independent convergence-order recomputation ---")
    cs = rm.get("convergence_study") if rm else None
    levels = (cs or {}).get("levels", [])
    if not cs:
        print("  (no convergence study to recompute)")
    elif len(levels) < 2:
        print("  (need >= 2 refinement levels to recompute an order; skipping)")
    else:
        # Norms are whatever the manifest recorded, not a fixed (L2, Linf) pair;
        # recompute an order only for a norm present at every level so each ratio
        # is defined. Compare against the assessment claim when one exists.
        norms = [n for n in sorted(levels[0].get("errors", {}))
                 if all(n in lv.get("errors", {}) for lv in levels)]
        # The manifest keys errors by norm only (one value per level, not per
        # field), but a numerical-assessment carries one convergence entry per
        # (field, norm). Group claims by norm: a norm mapping to a single claim is
        # checkable; a norm with several per-field claims is ambiguous here — the
        # flat manifest cannot say which field a recomputed order belongs to — so
        # skip it rather than compare against an arbitrary field. (Contract gap:
        # per-field errors in the manifest are deferred; see DECISIONS.md D22.)
        claims_by_norm = {}
        for c in (na or {}).get("convergence", []):
            claims_by_norm.setdefault(c["norm"], []).append(c["observed_order"])
        if not norms:
            print("  (no norm present at every level; skipping)")
        for norm in norms:
            print(f"  {norm}:")
            orders = []
            for i in range(1, len(levels)):
                e0, e1 = levels[i-1]["errors"][norm], levels[i]["errors"][norm]
                h0, h1 = levels[i-1]["h"], levels[i]["h"]
                # The order is undefined when an error is non-positive (log(0) or a
                # negative log argument) or the mesh scale repeats (log(h0/h1) == 0,
                # e.g. a temporal-refinement study holding spatial h fixed). Skip
                # such a pair rather than raise, matching the skip-rather-than-compare
                # approach used for the D22 per-field case above.
                if not (e0 > 0 and e1 > 0 and h0 > 0 and h1 > 0 and h0 != h1):
                    print(f"    h {h0:.5f}->{h1:.5f}: order undefined (skipped)")
                    continue
                p = math.log(e0/e1) / math.log(h0/h1)
                orders.append(p)
                print(f"    h {h0:.5f}->{h1:.5f}: order = {p:.3f}")
            if not orders:
                print(f"    (no comparable level pair for norm {norm!r}; skipping)")
                continue
            avg = sum(orders)/len(orders)
            claim_list = claims_by_norm.get(norm, [])
            if len(claim_list) == 1:
                good = abs(avg - claim_list[0]) < 0.05
                print(f"    mean {avg:.3f} vs assessment claim {claim_list[0]} -> [{'ok' if good else 'FAIL'}]")
                ok = ok and good
            elif len(claim_list) > 1:
                print(f"    mean {avg:.3f} ({len(claim_list)} per-field claims for norm {norm!r}; "
                      f"manifest errors are not keyed per field, skipping check)")
            else:
                print(f"    mean {avg:.3f} (no assessment claim to check)")
    return ok

print("using: self-contained subset validator")
ok = True

# case index (reuse registry) validates against its own schema
ci = load(ROOT / "components" / "case-index.json")
cis = load(CONTRACTS / "case-index.schema.json")
cerrs = []
validate(ci, cis, "", cerrs)
if cerrs:
    ok = False
    print("[FAIL] case-index.json")
    for e in cerrs[:12]:
        print(f"       {e}")
else:
    print(f"[ ok ] case-index.json ({len(ci['cases'])} case(s))")

# A worked study is any examples/ subdirectory holding a problem-spec.json
# (simulation lane) or, for the programming lane (D21, which emits no problem
# spec), a results-manifest.json — so plan-less studies are validated, not
# silently skipped.
studies = sorted(
    d for d in EXAMPLES.glob("*")
    if d.is_dir() and ((d / "problem-spec.json").is_file() or (d / "results-manifest.json").is_file())
)
if not studies:
    print("\nno worked studies under examples/ to validate")
for study in studies:
    # The D18 drift gate must report every study, so an unexpected error is
    # caught and recorded as a failure for that study rather than aborting the
    # remaining validations with a traceback.
    try:
        ok = check_study(study) and ok
    except Exception as e:
        ok = False
        print(f"\n### study: {study.name}")
        print(f"[FAIL] unexpected error: {type(e).__name__}: {e}")

print("\n=== DRY-RUN", "PASSED ===" if ok else "FAILED ===")
sys.exit(0 if ok else 1)
