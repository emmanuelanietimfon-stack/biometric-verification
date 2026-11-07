"""Reproducible, validated Monte-Carlo engine for a multimodal biometric simulator.

All biometric scores and detector probabilities are synthetic assumptions. No
real biometric samples, credentials, devices, or templates are processed.
"""
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd

TRAITS = ("fingerprint", "iris", "face")
ATTACKS = ("print", "dummy", "replay")
ATTACK_LABEL = {"print": "Printed photo / iris", "dummy": "Dummy finger", "replay": "HD replay"}
ATTACK_TRAIT = {"print": ("face", "iris"), "dummy": ("fingerprint",), "replay": ("face", "iris")}
TRAIT_PARAMS = {
    "fingerprint": dict(gen_max=.92, sd_g=.07, imp_mu=.25, sd_i=.10),
    "iris": dict(gen_max=.96, sd_g=.05, imp_mu=.20, sd_i=.08),
    "face": dict(gen_max=.86, sd_g=.09, imp_mu=.30, sd_i=.12),
}
N_BANDS = 3


@dataclass(frozen=True)
class Config:
    poor_share: float = .20
    target_far: float = .001
    n: int = 30000
    seed: int = 42
    spoof_fidelity: float = .92
    passive_print: float = .95
    passive_dummy: float = .90
    passive_replay: float = .55
    active_face_iris: float = .90
    fp_hw_sensor: bool = False
    fp_hw_detect: float = .90
    passive_false_reject: float = .004
    active_false_reject: float = .008
    borderline_margin: float = .06
    fallback_success: float = .98       # legitimate user's simulated PIN+OTP success
    fallback_attacker_success: float = .001  # simulated attacker guess/compromise probability
    w_fingerprint: float = .35
    w_iris: float = .40
    w_face: float = .25

    def __post_init__(self):
        bounded = ("poor_share", "target_far", "spoof_fidelity", "passive_print", "passive_dummy",
                   "passive_replay", "active_face_iris", "fp_hw_detect", "passive_false_reject",
                   "active_false_reject", "fallback_success", "fallback_attacker_success")
        for name in bounded:
            v = getattr(self, name)
            if not np.isfinite(v) or not 0 <= v <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not 0 <= self.borderline_margin <= 1:
            raise ValueError("borderline_margin must be between 0 and 1")
        if self.n < 100 or self.n > 2_000_000:
            raise ValueError("n must be between 100 and 2,000,000")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if min(self.w_fingerprint, self.w_iris, self.w_face) < 0:
            raise ValueError("trait weights cannot be negative")
        if sum((self.w_fingerprint, self.w_iris, self.w_face)) <= 0:
            raise ValueError("at least one trait weight must be positive")

    def weights(self):
        raw = dict(fingerprint=self.w_fingerprint, iris=self.w_iris, face=self.w_face)
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}

    def passive(self, attack):
        return {"print": self.passive_print, "dummy": self.passive_dummy,
                "replay": self.passive_replay}[attack]

    def active(self, trait):
        if trait in ("iris", "face"):
            return self.active_face_iris
        return self.fp_hw_detect if self.fp_hw_sensor else 0.0


@dataclass(frozen=True)
class Spec:
    name: str
    score: str
    liveness: bool = False
    active: bool = True
    dynamic: bool = False
    fallback: bool = False


BASELINE_SPECS = [
    Spec("M1 Fingerprint only", "single"),
    Spec("M2 AND gate (FP+Iris)", "and"),
    Spec("M3 Plain weighted (3 traits)", "plain"),
    Spec("M4a Proposed (no fallback)", "qw", True, True, True, False),
    Spec("M4b Proposed + fallback", "qw", True, True, True, True),
]


def _validate_kind(kind):
    if kind not in ("genuine", "impostor", "spoof"):
        raise ValueError("kind must be genuine, impostor, or spoof")


def _quality(rng, n, poor_share):
    poor = rng.random(n) < poor_share
    q = np.where(poor, rng.beta(1.6, 5.0, n), rng.beta(6.0, 1.6, n))
    return np.clip(q, .02, 1.0)


def genuine(rng, n, poor_share):
    scores, quality = {}, {}
    for t, p in TRAIT_PARAMS.items():
        q = _quality(rng, n, poor_share)
        mu = p["gen_max"] * (.55 + .45 * q)
        scores[t] = np.clip(rng.normal(mu, p["sd_g"] * (1 + (1 - q))), 0, 1)
        quality[t] = q
    return scores, quality


def impostor(rng, n, poor_share):
    scores, quality = {}, {}
    for t, p in TRAIT_PARAMS.items():
        q = _quality(rng, n, poor_share)
        scores[t] = np.clip(rng.normal(p["imp_mu"], p["sd_i"] * (1 + .5 * (1 - q))), 0, 1)
        quality[t] = q
    return scores, quality


def spoof(rng, n, poor_share, spoofed, fidelity):
    gs, gq = genuine(rng, n, poor_share)
    ims, imq = impostor(rng, n, poor_share)
    scores, quality = {}, {}
    for t in TRAITS:
        if t in spoofed:
            scores[t], quality[t] = np.clip(gs[t] * fidelity, 0, 1), gq[t]
        else:
            scores[t], quality[t] = ims[t], imq[t]
    return scores, quality


def spoofed_traits(scenario, attack):
    if scenario not in ("full", "partial"):
        raise ValueError("scenario must be full or partial")
    return list(TRAITS) if scenario == "full" else (["fingerprint"] if attack == "dummy" else ["face"])


def liveness_flags(rng, cfg, n, attack, traits, active):
    if attack not in ATTACKS:
        raise ValueError(f"unknown attack: {attack}")
    flagged = np.zeros(n, dtype=bool)
    for trait in traits:
        attack_type = attack if trait in ATTACK_TRAIT[attack] else "print"
        passive = cfg.passive(attack_type)
        active_rate = cfg.active(trait) if active else 0.0
        combined = 1 - (1 - passive) * (1 - active_rate)
        flagged |= rng.random(n) < combined
    return flagged


def liveness_false_rejects(rng, cfg, n, active):
    flagged = np.zeros(n, dtype=bool)
    for trait in TRAITS:
        flagged |= rng.random(n) < cfg.passive_false_reject
        if active and (trait in ("iris", "face") or (trait == "fingerprint" and cfg.fp_hw_sensor)):
            flagged |= rng.random(n) < cfg.active_false_reject
    return flagged


def make_score_fn(cfg, kind):
    weights = cfg.weights()
    if kind == "single":
        return lambda s, q: s["fingerprint"]
    if kind == "and":
        return lambda s, q: np.minimum(s["fingerprint"], s["iris"])
    if kind == "plain":
        return lambda s, q: sum(weights[t] * s[t] for t in TRAITS)
    if kind == "qw":
        def quality_weighted(scores, quality):
            numerator = sum(weights[t] * quality[t] * scores[t] for t in TRAITS)
            denominator = sum(weights[t] * quality[t] for t in TRAITS)
            return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        return quality_weighted
    raise ValueError(f"unknown score type: {kind}")


def mean_quality(cfg, quality):
    weights = cfg.weights()
    return sum(weights[t] * quality[t] for t in TRAITS)


class Method:
    def __init__(self, cfg, spec, rng):
        self.cfg, self.spec, self.rng = cfg, spec, rng
        self.score_fn = make_score_fn(cfg, spec.score)
        self._calibrate()

    def _calibrate(self):
        cfg = self.cfg
        scores, quality = impostor(self.rng, max(cfg.n, 40000), cfg.poor_share)
        combined = self.score_fn(scores, quality)
        # A single global threshold is retained for comparison methods. Dynamic
        # bands use a safe fallback to the global threshold if a bin is sparse.
        self.global_threshold = float(np.quantile(combined, 1 - cfg.target_far, method="higher"))
        if self.spec.dynamic:
            qbar = mean_quality(cfg, quality)
            self.edges = np.unique(np.quantile(qbar, np.linspace(0, 1, N_BANDS + 1)[1:-1]))
            bands = np.digitize(qbar, self.edges)
            self.bthr = np.full(len(self.edges) + 1, self.global_threshold)
            for band in range(len(self.bthr)):
                values = combined[bands == band]
                if len(values) >= 100:
                    self.bthr[band] = np.quantile(values, 1 - cfg.target_far, method="higher")

    def threshold(self, quality, n):
        if self.spec.dynamic:
            return self.bthr[np.digitize(mean_quality(self.cfg, quality), self.edges)]
        return np.full(n, self.global_threshold)

    def decide(self, scores, quality, live_flag=None):
        fused = self.score_fn(scores, quality)
        threshold = self.threshold(quality, len(fused))
        live = np.zeros(len(fused), dtype=bool) if live_flag is None else np.asarray(live_flag, dtype=bool)
        # A failed liveness check is a hard rejection and can NEVER enter fallback.
        direct = (fused >= threshold) & ~live
        fallback = self.spec.fallback & (fused < threshold) & (fused >= threshold - self.cfg.borderline_margin) & ~live
        return direct, fallback, fused, threshold


def _fallback_pass(rng, escalated, probability):
    return escalated & (rng.random(len(escalated)) < probability)


def evaluate(cfg, spec, rng=None):
    rng = rng or np.random.default_rng(cfg.seed)
    method = Method(cfg, spec, rng)
    n = cfg.n
    out = {"method": spec.name}

    scores, quality = genuine(rng, n, cfg.poor_share)
    live = liveness_false_rejects(rng, cfg, n, spec.active) if spec.liveness else None
    direct, escalated, _, _ = method.decide(scores, quality, live)
    fallback_ok = _fallback_pass(rng, escalated, cfg.fallback_success)
    out["FRR_%"] = 100 * (1 - (direct | fallback_ok).mean())
    out["FRR_no_fallback_%"] = 100 * (1 - direct.mean())
    out["Fallback_rate_%"] = 100 * escalated.mean()

    scores, quality = impostor(rng, n, cfg.poor_share)
    live = np.zeros(n, bool) if spec.liveness else None
    direct, escalated, _, _ = method.decide(scores, quality, live)
    attacker_fallback = _fallback_pass(rng, escalated, cfg.fallback_attacker_success)
    out["FAR_%"] = 100 * (direct | attacker_fallback).mean()

    for scenario in ("partial", "full"):
        rates = []
        for attack in ATTACKS:
            traits = spoofed_traits(scenario, attack)
            scores, quality = spoof(rng, n, cfg.poor_share, traits, cfg.spoof_fidelity)
            live = liveness_flags(rng, cfg, n, attack, traits, spec.active) if spec.liveness else None
            direct, escalated, _, _ = method.decide(scores, quality, live)
            attacker_fallback = _fallback_pass(rng, escalated, cfg.fallback_attacker_success)
            rate = 100 * (direct | attacker_fallback).mean()
            out[f"SAR_{scenario}_{attack}_%"] = rate
            rates.append(rate)
        out[f"SAR_{scenario}_avg_%"] = float(np.mean(rates))
    return out


def run_methods(cfg, specs):
    rows = []
    # Independent, deterministic random streams make results repeatable and
    # prevent a method's random-number consumption changing another method.
    for index, spec in enumerate(specs):
        rows.append(evaluate(cfg, spec, np.random.default_rng(np.random.SeedSequence([cfg.seed, index]))))
    return pd.DataFrame(rows).round(4)


def quality_sweep(cfg, specs, shares=(0, .1, .2, .3, .4, .5)):
    rows = []
    small = replace(cfg, n=min(cfg.n, 15000))
    for share in shares:
        current = replace(small, poor_share=float(share))
        frame = run_methods(current, specs)
        for _, result in frame.iterrows():
            rows.append({"poor_share_%": int(share * 100), "method": result["method"],
                         "FRR_%": result["FRR_%"], "FAR_%": result["FAR_%"]})
    return pd.DataFrame(rows).round(4)


def ablation(cfg):
    variants = [
        Spec("Full design", "qw", True, True, True, False),
        Spec("No liveness", "qw", False, True, True, False),
        Spec("Passive liveness only", "qw", True, False, True, False),
        Spec("No quality weighting", "plain", True, True, True, False),
        Spec("No dynamic threshold", "qw", True, True, False, False),
        Spec("Full design + fallback", "qw", True, True, True, True),
    ]
    return run_methods(cfg, variants)


def det_curves(cfg):
    rng = np.random.default_rng(cfg.seed)
    genuine_scores, genuine_q = genuine(rng, cfg.n, cfg.poor_share)
    impostor_scores, impostor_q = impostor(rng, cfg.n, cfg.poor_share)
    thresholds = np.linspace(0, 1, 801)
    curves, eers = {}, {}
    for name, kind in (("Fingerprint only", "single"), ("AND gate (FP+Iris)", "and"),
                       ("Plain weighted", "plain"), ("Quality-weighted", "qw")):
        fn = make_score_fn(cfg, kind)
        genuine_match, impostor_match = fn(genuine_scores, genuine_q), fn(impostor_scores, impostor_q)
        far = np.array([(impostor_match >= threshold).mean() for threshold in thresholds])
        frr = np.array([(genuine_match < threshold).mean() for threshold in thresholds])
        index = int(np.argmin(np.abs(far - frr)))
        curves[name] = (far, frr)
        eers[name] = 100 * (far[index] + frr[index]) / 2
    return curves, eers


def single_attempt(cfg, spec, kind, qualities=None, attack="print", scenario="full", seed=None):
    _validate_kind(kind)
    if attack not in ATTACKS:
        raise ValueError(f"unknown attack: {attack}")
    if qualities:
        for trait, value in qualities.items():
            if trait not in TRAITS or not 0 <= float(value) <= 1:
                raise ValueError(f"invalid quality for {trait}")
    rng = np.random.default_rng(seed)
    method = Method(cfg, spec, np.random.default_rng(cfg.seed))
    if kind == "genuine":
        scores, quality = genuine(rng, 1, cfg.poor_share)
        spoofed = []
    elif kind == "impostor":
        scores, quality = impostor(rng, 1, cfg.poor_share)
        spoofed = []
    else:
        spoofed = spoofed_traits(scenario, attack)
        scores, quality = spoof(rng, 1, cfg.poor_share, spoofed, cfg.spoof_fidelity)

    if qualities:
        for trait, value in qualities.items():
            p = TRAIT_PARAMS[trait]
            q = float(value)
            quality[trait] = np.array([q])
            if kind == "genuine" or (kind == "spoof" and trait in spoofed):
                mu = p["gen_max"] * (.55 + .45 * q)
                if kind == "spoof":
                    mu *= cfg.spoof_fidelity
                scores[trait] = np.clip(rng.normal(mu, p["sd_g"] * (1 + (1 - q)), 1), 0, 1)
            else:
                scores[trait] = np.clip(rng.normal(p["imp_mu"], p["sd_i"] * (1 + .5 * (1 - q)), 1), 0, 1)

    live = np.zeros(1, bool)
    if spec.liveness:
        live = (liveness_flags(rng, cfg, 1, attack, spoofed, spec.active) if kind == "spoof"
                else liveness_false_rejects(rng, cfg, 1, spec.active))
    direct, escalated, fused, threshold = method.decide(scores, quality, live if spec.liveness else None)
    fallback_probability = cfg.fallback_success if kind == "genuine" else cfg.fallback_attacker_success
    fallback_passed = bool(_fallback_pass(rng, escalated, fallback_probability)[0])
    accepted = bool(direct[0] or fallback_passed)
    if direct[0]:
        decision = "ACCEPTED"
    elif fallback_passed:
        decision = "ACCEPTED via simulated PIN + OTP fallback"
    elif escalated[0]:
        decision = "REJECTED: fallback verification failed"
    elif live[0] and spec.liveness:
        decision = "REJECTED: liveness check failed"
    else:
        decision = "REJECTED: match score below threshold"
    return {
        "scores": {t: float(scores[t][0]) for t in TRAITS},
        "quality": {t: float(quality[t][0]) for t in TRAITS},
        "liveness_flagged": bool(live[0]),
        "fused_score": float(fused[0]), "threshold": float(threshold[0]),
        "fallback_required": bool(escalated[0]), "fallback_passed": fallback_passed,
        "decision": decision, "accepted": accepted,
    }
