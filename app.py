"""
Interactive Biometric Security Simulator
Run from VS Code terminal:   streamlit run app.py

Tests the design from the literature review: quality-aware multimodal
authentication with liveness detection, compared against the baselines.
All scores and liveness rates are ASSUMED - this is a simulation, not a test
on real biometric devices.
"""
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import engine as E

st.set_page_config(page_title="Biometric Security Simulator", layout="wide")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COLORS = ["#8c8c8c", "#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b"]

st.title("Biometric Security Simulator")
st.caption("Quality-aware multimodal authentication with liveness detection. "
           "Change the settings on the left and watch the results update.")

# ------------------------------------------------------------------ sidebar
sb = st.sidebar
sb.header("Settings")

with sb.expander("1. Conditions", expanded=True):
    poor = st.slider("Poor-quality samples (%)", 0, 60, 20, 5,
                     help="Injuries, bad lighting, blur, dirty sensors.") / 100
    far_pct = st.select_slider("Target FAR (impostors accepted)",
                               options=[0.01, 0.05, 0.1, 0.5, 1.0], value=0.1,
                               help="Every method is tuned to this same FAR so the comparison is fair.")
    n = st.select_slider("Trials per test", options=[5000, 10000, 20000, 30000, 60000], value=20000)
    seed = st.number_input("Random seed", 0, 99999, 42)

with sb.expander("2. Attacker"):
    fidelity = st.slider("Fake quality (how closely a fake copies the real trait)", 0.5, 1.0, 0.92, 0.01)

with sb.expander("3. Liveness detection"):
    p_print = st.slider("Passive: catches printed photo/iris", 0.0, 1.0, 0.95, 0.01)
    p_dummy = st.slider("Passive: catches dummy finger", 0.0, 1.0, 0.90, 0.01)
    p_replay = st.slider("Passive: catches HD replay", 0.0, 1.0, 0.55, 0.01)
    p_active = st.slider("Active challenge-response (face, iris)", 0.0, 1.0, 0.90, 0.01,
                         help="Blink / head-turn / pupil response.")
    fp_hw = st.checkbox("Add hardware liveness sensor to fingerprint (pulse / sweat)", False)
    fp_hw_p = st.slider("Fingerprint sensor detection", 0.0, 1.0, 0.90, 0.01, disabled=not fp_hw)
    fr_pass = st.slider("Genuine wrongly flagged: passive (per trait, %)", 0.0, 3.0, 0.4, 0.1) / 100
    fr_act = st.slider("Genuine wrongly flagged: active (per check, %)", 0.0, 3.0, 0.8, 0.1) / 100

with sb.expander("4. Fallback (PIN + OTP)"):
    margin = st.slider("Borderline margin (score points below threshold)", 0.0, 0.2, 0.06, 0.01)
    fb_ok = st.slider("Genuine user completes fallback", 0.5, 1.0, 0.98, 0.01)
    fb_attack = st.slider("Attacker passes fallback (guess/compromise probability)", 0.0, 0.05, 0.001, 0.001,
                          help="A simulated probability, not a real PIN or OTP check.")

with sb.expander("5. Trait weights"):
    w_fp = st.slider("Fingerprint", 0.0, 1.0, 0.35, 0.05)
    w_ir = st.slider("Iris", 0.0, 1.0, 0.40, 0.05)
    w_fa = st.slider("Face", 0.0, 1.0, 0.25, 0.05)

with sb.expander("6. Build your own design", expanded=True):
    rule = st.selectbox("Scoring rule", ["Quality-weighted fusion (3 traits)",
                                         "Plain weighted fusion (3 traits)",
                                         "AND gate (FP + Iris)", "Fingerprint only"])
    c_live = st.checkbox("Liveness detection", True)
    c_active = st.checkbox("Active challenge-response", True, disabled=not c_live)
    c_dyn = st.checkbox("Dynamic threshold (per quality band)", True)
    c_fb = st.checkbox("Fallback (PIN + OTP)", True)

cfg = E.Config(poor_share=poor, target_far=far_pct / 100, n=int(n), seed=int(seed),
               spoof_fidelity=fidelity, passive_print=p_print, passive_dummy=p_dummy,
               passive_replay=p_replay, active_face_iris=p_active, fp_hw_sensor=fp_hw,
               fp_hw_detect=fp_hw_p, passive_false_reject=fr_pass, active_false_reject=fr_act,
               borderline_margin=margin, fallback_success=fb_ok, fallback_attacker_success=fb_attack,
               w_fingerprint=w_fp, w_iris=w_ir, w_face=w_fa)

score_kind = {"Quality-weighted fusion (3 traits)": "qw", "Plain weighted fusion (3 traits)": "plain",
              "AND gate (FP + Iris)": "and", "Fingerprint only": "single"}[rule]
custom = E.Spec("YOUR DESIGN", score_kind, liveness=c_live, active=c_active and c_live,
                dynamic=c_dyn, fallback=c_fb)
SPECS = E.BASELINE_SPECS + [custom]


@st.cache_data(show_spinner="Running simulation...")
def cached_methods(cfg, specs):
    return E.run_methods(cfg, list(specs))


@st.cache_data(show_spinner="Running quality sweep (a few seconds)...")
def cached_sweep(cfg, specs):
    return E.quality_sweep(cfg, list(specs))


@st.cache_data(show_spinner="Running ablation...")
def cached_ablation(cfg):
    return E.ablation(cfg)


@st.cache_data(show_spinner="Computing DET curves...")
def cached_det(cfg):
    return E.det_curves(cfg)


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Compare methods", "Quality stress test", "Component ablation", "Try one login", "About"])

# ------------------------------------------------------------------ tab 1
with tab1:
    df = cached_methods(cfg, tuple(SPECS))
    mine = df[df.method == "YOUR DESIGN"].iloc[0]
    base = df[df.method == "M2 AND gate (FP+Iris)"].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Your design: genuine rejected (FRR)", f"{mine['FRR_%']:.2f}%",
              f"{mine['FRR_%'] - base['FRR_%']:+.2f} vs AND gate", delta_color="inverse")
    c2.metric("Impostors accepted (FAR)", f"{mine['FAR_%']:.3f}%")
    c3.metric("Spoofs accepted, one trait faked", f"{mine['SAR_partial_avg_%']:.2f}%",
              f"{mine['SAR_partial_avg_%'] - base['SAR_partial_avg_%']:+.2f} vs AND gate", delta_color="inverse")
    c4.metric("Spoofs accepted, all traits faked", f"{mine['SAR_full_avg_%']:.3f}%",
              f"{mine['SAR_full_avg_%'] - base['SAR_full_avg_%']:+.2f} vs AND gate", delta_color="inverse")

    show = df[["method", "FAR_%", "FRR_%", "FRR_no_fallback_%", "Fallback_rate_%",
               "SAR_partial_avg_%", "SAR_full_avg_%"]]
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button("Download results (CSV)", df.to_csv(index=False), "results.csv")

    ca, cb = st.columns(2)
    with ca:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.barh(df.method, df["FRR_%"], color=COLORS[:len(df)])
        ax.set_xlabel("Genuine users rejected, FRR (%)"); ax.invert_yaxis()
        ax.set_title("Genuine users rejected")
        st.pyplot(fig); plt.close(fig)
    with cb:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        w = 0.38; y = np.arange(len(df))
        ax.barh(y - w / 2, df["SAR_partial_avg_%"], w, label="One trait faked", color="#ff7f0e")
        ax.barh(y + w / 2, df["SAR_full_avg_%"], w, label="All traits faked", color="#d62728")
        ax.set_yticks(y); ax.set_yticklabels(df.method); ax.invert_yaxis()
        ax.set_xscale("symlog", linthresh=0.1); ax.legend(fontsize=7)
        ax.set_xlabel("Spoof acceptance (%, symlog)"); ax.set_title("Spoofs accepted")
        st.pyplot(fig); plt.close(fig)

    with st.expander("Spoof acceptance by attack type"):
        cols = ["method"] + [f"SAR_{s}_{a}_%" for s in ("partial", "full") for a in E.ATTACKS]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)

    with st.expander("Matching accuracy (DET curves, no liveness)"):
        curves, eers = cached_det(cfg)
        fig, ax = plt.subplots(figsize=(6, 4))
        for (name, (far, frr)), col in zip(curves.items(), COLORS):
            ax.plot(np.maximum(far, 1e-5) * 100, np.maximum(frr, 1e-5) * 100,
                    label=f"{name} (EER {eers[name]:.2f}%)", color=col)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-3, 50); ax.set_ylim(1e-3, 50)
        ax.set_xlabel("FAR (%)"); ax.set_ylabel("FRR (%)"); ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")
        st.pyplot(fig); plt.close(fig)

# ------------------------------------------------------------------ tab 2
with tab2:
    st.write("How does each method cope as sample quality gets worse? "
             "(Thresholds stay calibrated to the Conditions you set, not re-tuned per level.)")
    if st.button("Run quality stress test"):
        st.session_state["sweep"] = cached_sweep(cfg, tuple(SPECS))
    if "sweep" in st.session_state:
        sw = st.session_state["sweep"]
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, m in enumerate(sw.method.unique()):
            d = sw[sw.method == m]
            ax.plot(d["poor_share_%"], d["FRR_%"], marker="o", label=m, color=COLORS[i % len(COLORS)])
        ax.set_xlabel("Poor-quality samples (%)"); ax.set_ylabel("Genuine users rejected, FRR (%)")
        ax.legend(fontsize=7); ax.grid(alpha=.3)
        st.pyplot(fig); plt.close(fig)
        st.dataframe(sw.pivot(index="poor_share_%", columns="method", values="FRR_%"),
                     use_container_width=True)

# ------------------------------------------------------------------ tab 3
with tab3:
    st.write("Switch parts of the proposed design off, one at a time, to see what each contributes.")
    if st.button("Run ablation"):
        st.session_state["abl"] = cached_ablation(cfg)
    if "abl" in st.session_state:
        ab = st.session_state["abl"]
        st.dataframe(ab[["method", "FAR_%", "FRR_%", "Fallback_rate_%", "SAR_partial_avg_%", "SAR_full_avg_%"]],
                     use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        x = np.arange(len(ab)); w = 0.38
        ax.bar(x - w / 2, ab["FRR_%"], w, label="Genuine rejected (FRR)", color="#1f77b4")
        ax.bar(x + w / 2, ab["SAR_full_avg_%"], w, label="Spoofs accepted (all traits faked)", color="#d62728")
        ax.set_xticks(x); ax.set_xticklabels(ab.method, rotation=18, ha="right", fontsize=8)
        ax.set_ylabel("%"); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
        st.pyplot(fig); plt.close(fig)

# ------------------------------------------------------------------ tab 4
with tab4:
    st.write("Simulate a single login attempt and see every step of the decision.")
    c1, c2 = st.columns([1, 1])
    with c1:
        pick = st.selectbox("Method", [s.name for s in SPECS], index=len(SPECS) - 1)
        kind = st.radio("Who is trying to log in?",
                        ["Genuine user", "Impostor (different person)", "Spoof attack (fake trait)"])
        attack, scen = "print", "full"
        if kind.startswith("Spoof"):
            attack = st.selectbox("Attack", E.ATTACKS, format_func=lambda a: E.ATTACK_LABEL[a])
            scen = st.radio("Traits faked", ["full", "partial"],
                            format_func=lambda s: "All traits" if s == "full" else "One trait only", horizontal=True)
    with c2:
        st.write("Sample quality (1 = perfect, 0 = very poor)")
        qf = st.slider("Fingerprint quality", 0.05, 1.0, 0.85, 0.05)
        qi = st.slider("Iris quality", 0.05, 1.0, 0.85, 0.05)
        qa = st.slider("Face quality", 0.05, 1.0, 0.85, 0.05)
    if "attempt_seed" not in st.session_state:
        st.session_state["attempt_seed"] = 1
    if st.button("Run attempt"):
        st.session_state["attempt_seed"] = int(np.random.randint(0, 10**6))
    spec = next(s for s in SPECS if s.name == pick)
    k = {"Genuine user": "genuine", "Impostor (different person)": "impostor"}.get(kind, "spoof")
    r = E.single_attempt(cfg, spec, k, {"fingerprint": qf, "iris": qi, "face": qa},
                         attack, scen, st.session_state["attempt_seed"])
    tbl = pd.DataFrame({"Quality": r["quality"], "Match score": r["scores"]}).round(3)
    st.table(tbl)
    if spec.liveness:
        st.write("Liveness check: " + ("**flagged as FAKE**" if r["liveness_flagged"] else "passed (looks live)"))
    else:
        st.write("Liveness check: none (this method has no liveness detection)")
    st.write(f"Fused score **{r['fused_score']:.3f}** vs threshold **{r['threshold']:.3f}**")
    (st.success if r["accepted"] else st.error)(f"Decision: {r['decision']}")
    st.caption("Press Run attempt repeatedly. A failed liveness check is a hard rejection and cannot be rescued by fallback. PIN + OTP is modeled probabilistically; no real credentials or OTP messages are used.")

# ------------------------------------------------------------------ tab 5
with tab5:
    st.markdown("""
**What this is.** A simulation of the design proposed in the literature review: multimodal
(fingerprint, iris, face) authentication with image-quality-based liveness detection,
quality-weighted fusion, a dynamic threshold and a PIN + OTP fallback.

**Methods compared.**
M1 fingerprint only. M2 fingerprint AND iris (AND gate, Mustafa et al.). M3 three traits, plain weighted.
M4a proposed design. M4b proposed + fallback. YOUR DESIGN is whatever you build in the sidebar.

**Fair comparison.** Every method's threshold is tuned to the same FAR, so differences in
rejections and spoof acceptance are real differences between methods.

**Terms.** FAR: impostors accepted. FRR: genuine users rejected. SAR: spoof acceptance rate.
EER: where FAR equals FRR.

**Limits.** All scores, quality values and liveness detection rates are assumed, not measured.
Results show how the decision rules behave under controlled conditions. Confirm with real data
(CASIA, FVC2004, LivDet) before claiming real-world performance.
""")
