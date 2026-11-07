"""
validate.py - quick self-check of the simulation engine.
Run in PowerShell / VS Code terminal:   python validate.py
Exit code 0 = all checks passed.
"""
import sys
from dataclasses import replace
import engine as E

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


cfg = E.Config(n=30000, seed=42)
df = E.run_methods(cfg, E.BASELINE_SPECS).set_index("method")

# 1. Fair comparison: every method calibrated to ~ the same FAR (0.1%)
far_ok = all(0.03 <= df.loc[m, "FAR_%"] <= 0.25 for m in df.index)
check("FAR close to 0.1% target for all methods", far_ok, str(df["FAR_%"].to_dict()))

# 2. Multimodal beats single trait on genuine rejections
check("Fingerprint-only rejects more genuine users than 3-trait fusion",
      df.loc["M1 Fingerprint only", "FRR_%"] > df.loc["M3 Plain weighted (3 traits)", "FRR_%"])

# 3. Liveness stops full spoofing; baselines without liveness do not
check("Baselines accept most all-trait spoofs (>50%)",
      all(df.loc[m, "SAR_full_avg_%"] > 50 for m in
          ["M1 Fingerprint only", "M2 AND gate (FP+Iris)", "M3 Plain weighted (3 traits)"]))
check("Proposed design accepts almost no all-trait spoofs (<0.5%)",
      df.loc["M4a Proposed (no fallback)", "SAR_full_avg_%"] < 0.5)

# 4. Fallback lowers genuine rejections and never raises spoof acceptance
a, b = df.loc["M4a Proposed (no fallback)"], df.loc["M4b Proposed + fallback"]
check("Fallback lowers FRR", b["FRR_%"] < a["FRR_%"])
check("Fallback does not raise spoof acceptance (within noise)",
      b["SAR_full_avg_%"] <= a["SAR_full_avg_%"] + 0.05)

# 5. AND gate degrades as quality worsens
lo = E.evaluate(replace(cfg, poor_share=0.0, n=20000), E.BASELINE_SPECS[1])["FRR_%"]
hi = E.evaluate(replace(cfg, poor_share=0.5, n=20000), E.BASELINE_SPECS[1])["FRR_%"]
check("AND-gate FRR rises with poor-quality share", hi > lo + 1, f"0%: {lo:.2f}  50%: {hi:.2f}")

# 6. Hardware fingerprint sensor reduces one-trait (dummy finger) spoofing
base = E.evaluate(cfg, E.BASELINE_SPECS[3])["SAR_partial_dummy_%"]
hw = E.evaluate(replace(cfg, fp_hw_sensor=True), E.BASELINE_SPECS[3])["SAR_partial_dummy_%"]
check("Fingerprint hardware sensor cuts dummy-finger acceptance", hw < base, f"{base:.2f} -> {hw:.2f}")

# 7. Same seed gives same answer (reproducible)
r1 = E.run_methods(cfg, E.BASELINE_SPECS)
r2 = E.run_methods(cfg, E.BASELINE_SPECS)
check("Reproducible with the same seed", r1.equals(r2))

# 8. Single-attempt walkthrough runs and returns a decision
r = E.single_attempt(cfg, E.BASELINE_SPECS[4], "genuine",
                     {"fingerprint": .8, "iris": .8, "face": .8}, seed=1)
check("single_attempt returns a decision", "decision" in r and isinstance(r["accepted"], bool))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
