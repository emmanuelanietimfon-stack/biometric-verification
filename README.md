# Multimodal Biometric Security Simulator

A reproducible Python/Streamlit research simulator for comparing fingerprint, iris and face authentication strategies. It models biometric score distributions, sample quality, multimodal score fusion, dynamic thresholds, spoof/liveness checks, a probabilistic PIN+OTP fallback, and common performance metrics.

> **Scope and safety:** this is a simulation, not a deployed identity system. It does not capture biometric images, communicate with sensors, enroll people, store biometric templates, validate real PINs/OTPs, create accounts, or issue access tokens. All score distributions and detector probabilities are configurable assumptions. Do not use its output as proof of real-world biometric accuracy or security.

## Features

- Five reference methods plus a configurable custom method.
- Fingerprint-only, FP+iris AND gate, plain weighted fusion and quality-weighted fusion.
- Quality-aware thresholds with sparse-bin safeguards.
- Passive and active liveness detection for print, dummy-finger and replay scenarios.
- Hard rejection on a failed liveness check; a liveness failure cannot be routed to fallback.
- Simulated PIN+OTP fallback with separate legitimate-user and attacker success probabilities.
- FAR, FRR, fallback rate, spoof acceptance (SAR), DET curves and EER estimates.
- Quality stress testing, component ablation and a step-by-step single-attempt view.
- Deterministic runs from a configurable random seed, CSV export and automated tests.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard and interactive controls |
| `engine.py` | Validated simulation model and metrics |
| `test_engine.py` | Automated regression and safety tests |
| `requirements.txt` | Runtime and test dependencies |
| `.gitignore` | Excludes local environments, caches and generated outputs |
| `.vscode/tasks.json` | VS Code run task |

## Run on Windows (VS Code / PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal, then activate again. Open the local URL printed by Streamlit (normally `http://localhost:8501`). Stop with `Ctrl+C`.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Run tests

```bash
python -m unittest -v
```

## Metrics

- **FAR**: impostor attempts accepted, including the configured simulated fallback-pass probability.
- **FRR**: genuine attempts rejected after any simulated fallback.
- **FRR without fallback**: genuine attempts not accepted directly by the biometric decision.
- **Fallback rate**: attempts routed to the simulated PIN+OTP step.
- **SAR**: spoof attempts accepted, including any simulated attacker fallback success.
- **EER**: approximate equal-error rate from the DET curve.

Rates are Monte-Carlo estimates and can vary with sample count and seed. Very low target FAR values require large sample counts to estimate reliably. The calibration quantile is not a guarantee that the observed test FAR will equal the target exactly.

## Modeling assumptions and research use

The score distributions and liveness probabilities are illustrative parameters in `engine.py`. Change them only with documented justification. For empirical validation, obtain appropriately licensed datasets and follow their protocols (for example, relevant fingerprint, iris, face and presentation-attack datasets). Keep training/calibration and evaluation partitions separate, report confidence intervals, and document dataset versions, sensor conditions, demographics where ethically and legally appropriate, and threat models. Never claim that assumed distributions demonstrate performance on real people or devices.

## Security boundary

No actual secrets or biometric identifiers are collected. The fallback is a probability model only. A production authentication service would require a separate threat model and implementation covering sensor trust, secure enrollment, template protection, privacy/consent, anti-replay, credential verification, rate limiting, secure transport, audit controls, recovery and token/session lifecycle. Those capabilities are intentionally outside this simulator.
