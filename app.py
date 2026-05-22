"""
app.py — Live Demo Hate Speech Detector (BLOOM-560m MTL v2)
============================================================
Futuristic Glassmorphism UI · GSAP 3 Animations · Dark Cyberpunk Theme

Jalankan:
    python app.py --checkpoint models/main_bloom_mtl_v2/best_model.pt
                  --bloom     models/bloom-560m-local
                  --threshold models/threshold_info_v2.json

BUG FIXES (vs original):
  1. SCRIPTS_HTML double-boot: added `let _booted = false` guard so the
     GSAP/particle init never runs twice even when both the 'load' event
     AND the 1200 ms fallback setTimeout fire.
  2. Button-hover flicker: replaced bubbling mouseover/mouseout with
     non-bubbling mouseenter/mouseleave on the event listeners.
  3. requirements.txt: added Pillow (used by lime_viz.fig_to_pil) and
     gdown (download helper).
  4. inference.py: I-TOX word_bio upgrade logic made explicit + in_tox
     state variable for correct span reconstruction.
"""

import argparse
import os
import sys
import torch
import gradio as gr

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

from model_utils import load_model, load_tokenizer, load_threshold
from inference   import (predict_text, build_highlighted_html,
                          build_bio_table, build_verdict_html)
from lime_viz    import run_lime_explanation, fig_to_pil

# ══════════════════════════════════════════════════════════════════════════
#  Defaults
# ══════════════════════════════════════════════════════════════════════════
DEFAULT_CHECKPOINT     = os.path.join(_DEMO_DIR, "models", "main_bloom_mtl_v2", "best_model.pt")
DEFAULT_BLOOM_PATH     = os.path.join(_DEMO_DIR, "models", "bloom-560m-local")
DEFAULT_THRESHOLD_JSON = os.path.join(_DEMO_DIR, "models", "threshold_info_v2.json")
DEFAULT_LIME_SAMPLES   = 200

_STATE = {
    "model":     None,
    "tokenizer": None,
    "threshold": 0.4429,
    "device":    None,
    "lime_n":    DEFAULT_LIME_SAMPLES,
    "ready":     False,
}


# ══════════════════════════════════════════════════════════════════════════
#  Initialization
# ══════════════════════════════════════════════════════════════════════════
def _initialize(checkpoint: str, bloom_path: str,
                threshold_json: str, lime_samples: int) -> None:
    _STATE["lime_n"] = lime_samples
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _STATE["device"] = device
    print(f"\n[App] Device: {device}")
    if device.type == "cuda":
        print(f"[App] GPU  : {torch.cuda.get_device_name(0)}")
        print(f"[App] VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    _STATE["tokenizer"] = load_tokenizer(bloom_path)
    _STATE["model"]     = load_model(checkpoint, bloom_path, device)
    _STATE["threshold"] = load_threshold(threshold_json)
    _STATE["ready"]     = True
    print("[App] ✅ Model siap. Buka browser di http://localhost:7860")


# ══════════════════════════════════════════════════════════════════════════
#  Callbacks
# ══════════════════════════════════════════════════════════════════════════
def fn_predict(text: str):
    if not _STATE["ready"]:
        err = "<p style='color:#ff4444;font-family:JetBrains Mono,monospace;'>[ ERROR: Model belum siap ]</p>"
        return err, err, [["—", "—", "—", "—"]], "Model belum siap"
    if not text or not text.strip():
        warn = "<p style='color:rgba(0,245,255,0.4);font-style:italic;font-family:JetBrains Mono,monospace;font-size:0.9em;'>[ Masukkan teks untuk dianalisis... ]</p>"
        return warn, warn, [["—", "—", "—", "—"]], "—"

    result = predict_text(
        text      = text,
        tokenizer = _STATE["tokenizer"],
        model     = _STATE["model"],
        device    = _STATE["device"],
        threshold = _STATE["threshold"],
    )
    verdict_html   = build_verdict_html(
        pred         = result["pred"],
        prob_hate    = result["prob_hate"],
        prob_nonhate = result["prob_nonhate"],
        threshold    = _STATE["threshold"],
        spans        = result["spans"],
    )
    highlight_html = build_highlighted_html(result["words"], result["word_bio"])
    bio_table      = build_bio_table(result["words"], result["word_bio"])
    status_txt     = (
        f"▶  {result['pred']}  ·  P(hate)={result['prob_hate']:.4f}"
        f"  ·  {len(result['spans'])} toxic span terdeteksi"
    )
    return verdict_html, highlight_html, bio_table, status_txt


def fn_lime(text: str, n_samples: int):
    if not _STATE["ready"]:
        return None, "[ ERROR: Model belum siap ]"
    if not text or not text.strip():
        return None, "[ WARN: Masukkan teks terlebih dahulu ]"

    n_samples = max(50, min(int(n_samples), 1000))

    result = predict_text(
        text      = text,
        tokenizer = _STATE["tokenizer"],
        model     = _STATE["model"],
        device    = _STATE["device"],
        threshold = _STATE["threshold"],
    )
    actual_label = 1 if result["pred"] == "HATE SPEECH" else 0

    fig, contributions, pred_label, confidence = run_lime_explanation(
        text         = text,
        tokenizer    = _STATE["tokenizer"],
        model        = _STATE["model"],
        device       = _STATE["device"],
        actual_label = actual_label,
        num_features = 10,
        num_samples  = n_samples,
    )
    if fig is None:
        return None, "[ WARN: LIME gagal — teks terlalu pendek? ]"

    pil_img = fig_to_pil(fig)
    top     = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:3]
    top_txt = "  ·  ".join(f'"{w}" ({v:+.4f})' for w, v in top)
    summary = (
        f"▶  Prediksi: {pred_label} ({confidence:.2%})  ·  Sampel: {n_samples}\n"
        f"▶  Top-3: {top_txt}\n"
        f"⚠   Interpretasi bersifat KUALITATIF (BPE vs word-level mismatch)."
    )
    return pil_img, summary


# ══════════════════════════════════════════════════════════════════════════
#  Example texts
# ══════════════════════════════════════════════════════════════════════════
EXAMPLES = [
    ["user jablay. pacar gua balikin dulu bangsat. gua mau belajar gak jadi jadi",
     "ID — Hate Speech"],
    ["dan setelah blokir jokowi saya harap ada lengserkan jokowi",
     "ID — Hate Speech (Politis)"],
    ["presiden joko widodo jokowi masuk daftar the muslim. kepala staf menyebut predikat itu.",
     "ID — Non-Hate"],
    ["kill those goat fuckers!",
     "EN — Hate Speech (Eksplisit)"],
    ["the fact that you are rationalising the institutionalised rape slavery murder of isis is mind boggling",
     "EN — Hate Speech (Implisit)"],
    ["this loser can hate america on twitter and i get suspended for calling lebron james what he is ghetto trash",
     "EN — Non-Hate (Ambigu)"],
    ["orang tionghoa tuh honestly always cuma mikirin keuntungan kelompok sendiri no sense of community",
     "Code-Mixed — Hate Speech"],
    ["kafir tuh musuh yang harus kita waspadai they work against our community from the inside",
     "Code-Mixed — Hate Speech"],
    ["mrt baru finally nyambung ke stasiun dekat kantor honestly saves me so much time",
     "Code-Mixed — Non-Hate"],
    ["your prompt response to emails is very professional sangat menghargai efisiensimu",
     "Code-Mixed — Non-Hate"],
]


# ══════════════════════════════════════════════════════════════════════════
#  Custom CSS — Futuristic Glassmorphism (Dark Cyberpunk)
# ══════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Syne:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Variables ── */
:root {
    --neon-cyan:    #00f5ff;
    --neon-purple:  #bf5fff;
    --neon-red:     #ff2d55;
    --neon-green:   #00ff88;
    --neon-amber:   #ffaa00;
    --bg-deep:      #050a18;
    --bg-card:      rgba(8, 14, 30, 0.75);
    --glass-border: rgba(0, 245, 255, 0.12);
    --glass-blur:   blur(24px);
    --text-1:       #dceeff;
    --text-2:       rgba(220, 238, 255, 0.55);
    --text-3:       rgba(220, 238, 255, 0.3);
    --font-display: 'Orbitron', monospace;
    --font-body:    'Syne', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
}

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }

body {
    background: var(--bg-deep) !important;
    font-family: var(--font-body) !important;
    color: var(--text-1) !important;
    overflow-x: hidden;
}

/* ── Background aurora ── */
.gradio-container {
    background: var(--bg-deep) !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
    position: relative !important;
}
.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 40% at 15% 55%, rgba(0,245,255,0.04) 0%, transparent 70%),
        radial-gradient(ellipse 50% 40% at 85% 20%, rgba(191,95,255,0.05) 0%, transparent 70%),
        radial-gradient(ellipse 40% 30% at 50% 90%, rgba(255,45,85,0.03) 0%, transparent 70%);
    pointer-events: none;
    z-index: -1;
    animation: aurora 12s ease-in-out infinite alternate;
}
@keyframes aurora {
    0%   { opacity: 0.6; transform: scale(1); }
    100% { opacity: 1;   transform: scale(1.05); }
}

/* ── Cyber grid overlay ── */
.gradio-container::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,245,255,0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,245,255,0.015) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: -1;
}

/* ── Glass card ── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    box-shadow:
        0 8px 40px rgba(0,0,0,0.5),
        0 0 0 1px rgba(0,245,255,0.04),
        inset 0 1px 0 rgba(255,255,255,0.04);
}

/* ── Tabs ── */
.tabs { background: transparent !important; border: none !important; }
.tab-nav {
    background: rgba(5, 10, 24, 0.85) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(0,245,255,0.12) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 2px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
    margin-bottom: 16px !important;
}
.tab-nav button {
    font-family: var(--font-display) !important;
    font-size: 0.72em !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    color: var(--text-2) !important;
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 22px !important;
    transition: all 0.25s ease !important;
    position: relative !important;
    overflow: hidden !important;
    text-transform: uppercase !important;
}
.tab-nav button.selected,
.tab-nav button[aria-selected="true"] {
    color: var(--neon-cyan) !important;
    background: rgba(0,245,255,0.08) !important;
    box-shadow: 0 0 24px rgba(0,245,255,0.15), inset 0 0 0 1px rgba(0,245,255,0.15) !important;
}
.tab-nav button:hover:not(.selected) {
    color: var(--text-1) !important;
    background: rgba(255,255,255,0.04) !important;
}

/* ── Textbox ── */
.gr-textbox, label > textarea, label > input[type="text"] {
    background: rgba(2, 6, 18, 0.8) !important;
    border: 1px solid rgba(0,245,255,0.18) !important;
    border-radius: 12px !important;
    color: var(--text-1) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.88em !important;
    line-height: 1.7 !important;
    padding: 12px 16px !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
    caret-color: var(--neon-cyan) !important;
}
label > textarea:focus, label > input:focus {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 0 1px rgba(0,245,255,0.25), 0 0 24px rgba(0,245,255,0.1) !important;
    outline: none !important;
}
label > textarea::placeholder { color: var(--text-3) !important; }

/* ── Labels ── */
label > span, .gr-block > label > span,
fieldset > legend, .label-wrap span {
    font-family: var(--font-mono) !important;
    font-size: 0.72em !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    color: rgba(0,245,255,0.6) !important;
    margin-bottom: 6px !important;
    display: block !important;
}

/* ── Primary button ── */
button.primary, button[class*="primary"], .gr-button-primary {
    font-family: var(--font-display) !important;
    font-size: 0.75em !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    background: linear-gradient(135deg, rgba(0,180,220,0.9), rgba(0,100,255,0.9)) !important;
    border: 1px solid rgba(0,245,255,0.5) !important;
    border-radius: 10px !important;
    color: #fff !important;
    padding: 13px 28px !important;
    box-shadow: 0 4px 20px rgba(0,180,255,0.35), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    transition: all 0.25s ease !important;
    position: relative !important;
    overflow: hidden !important;
}
button.primary::after, button[class*="primary"]::after {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 60%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
    transition: left 0.5s ease;
    pointer-events: none;
}
button.primary:hover::after, button[class*="primary"]:hover::after { left: 150%; }
button.primary:hover, button[class*="primary"]:hover {
    box-shadow: 0 6px 30px rgba(0,220,255,0.5), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    transform: translateY(-2px) !important;
    border-color: rgba(0,245,255,0.8) !important;
}
button.primary:active, button[class*="primary"]:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 10px rgba(0,180,255,0.3) !important;
}

/* ── Secondary button ── */
button.secondary, button[class*="secondary"], .gr-button-secondary {
    font-family: var(--font-display) !important;
    font-size: 0.72em !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    background: rgba(191,95,255,0.1) !important;
    border: 1px solid rgba(191,95,255,0.35) !important;
    border-radius: 10px !important;
    color: var(--neon-purple) !important;
    padding: 13px 28px !important;
    box-shadow: 0 4px 16px rgba(191,95,255,0.15) !important;
    transition: all 0.25s ease !important;
    overflow: hidden !important;
    position: relative !important;
}
button.secondary:hover, button[class*="secondary"]:hover {
    background: rgba(191,95,255,0.18) !important;
    box-shadow: 0 6px 28px rgba(191,95,255,0.35) !important;
    transform: translateY(-2px) !important;
    border-color: rgba(191,95,255,0.7) !important;
}

/* ── Status textbox ── */
.status-box textarea {
    font-family: var(--font-mono) !important;
    font-size: 0.8em !important;
    color: var(--neon-cyan) !important;
    background: rgba(0,245,255,0.04) !important;
    border-color: rgba(0,245,255,0.12) !important;
    letter-spacing: 0.3px !important;
}

/* ── Dataframe / table ── */
.gr-dataframe table, table.gr-samples-table {
    background: transparent !important;
    border-collapse: collapse !important;
    width: 100% !important;
}
.gr-dataframe {
    background: rgba(2,6,18,0.6) !important;
    border: 1px solid rgba(0,245,255,0.1) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
.gr-dataframe thead tr th {
    background: rgba(0,245,255,0.07) !important;
    color: var(--neon-cyan) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72em !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid rgba(0,245,255,0.15) !important;
    border-right: 1px solid rgba(0,245,255,0.06) !important;
}
.gr-dataframe tbody tr {
    border-bottom: 1px solid rgba(255,255,255,0.04) !important;
    transition: background 0.15s ease !important;
}
.gr-dataframe tbody tr:hover { background: rgba(0,245,255,0.04) !important; }
.gr-dataframe tbody td {
    color: var(--text-1) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82em !important;
    padding: 8px 14px !important;
    border-right: 1px solid rgba(255,255,255,0.04) !important;
}

/* ── Slider ── */
.gr-slider input[type=range] {
    accent-color: var(--neon-purple) !important;
    cursor: pointer;
}

/* ── Image ── */
.gr-image, .gr-image img {
    border-radius: 12px !important;
    border: 1px solid rgba(191,95,255,0.2) !important;
}

/* ── Accordion ── */
details {
    background: rgba(2,6,18,0.6) !important;
    border: 1px solid rgba(0,245,255,0.1) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin-top: 12px !important;
}
details summary {
    color: rgba(0,245,255,0.65) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8em !important;
    cursor: pointer !important;
    letter-spacing: 0.5px !important;
    list-style: none !important;
}
details summary::-webkit-details-marker { display: none; }
details summary::before {
    content: '▶ ';
    color: var(--neon-cyan);
    font-size: 0.7em;
}
details[open] summary::before { content: '▼ '; }

/* ── Markdown / prose ── */
.gr-markdown h3 {
    font-family: var(--font-display) !important;
    font-size: 0.8em !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    color: var(--neon-cyan) !important;
    text-transform: uppercase !important;
    margin: 12px 0 6px !important;
}
.gr-markdown p, .gr-markdown li {
    color: var(--text-2) !important;
    font-family: var(--font-body) !important;
    font-size: 0.88em !important;
    line-height: 1.7 !important;
}

/* ── Examples ── */
.gr-samples-table { background: rgba(0,0,0,0.25) !important; }
.gr-samples-table td {
    color: var(--text-2) !important;
    font-size: 0.82em !important;
    cursor: pointer !important;
}
.gr-samples-table tr:hover td { color: var(--neon-cyan) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); }
::-webkit-scrollbar-thumb { background: rgba(0,245,255,0.25); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,245,255,0.45); }

/* ── Block background overrides ── */
.gr-block, .gr-form, .gr-box, .wrap.default,
[class*="block"], .border-none { background: transparent !important; }

/* ── Glow divider ── */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,245,255,0.25), transparent);
    margin: 10px 0;
    border: none;
}

/* ── Loading bar (top of page while predicting) ── */
#hs-loader-bar {
    position: fixed;
    top: 0; left: 0;
    height: 2px;
    width: 0%;
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-purple));
    box-shadow: 0 0 10px var(--neon-cyan);
    z-index: 10000;
    transition: width 0.1s linear;
    pointer-events: none;
}

/* ── Keyframes ── */
@keyframes scanline {
    0%   { transform: translateY(-100%); }
    100% { transform: translateY(600%); }
}
@keyframes neonPulse {
    0%, 100% {
        filter: drop-shadow(0 0 8px rgba(0,245,255,0.4))
                drop-shadow(0 0 20px rgba(0,245,255,0.2));
    }
    50% {
        filter: drop-shadow(0 0 16px rgba(0,245,255,0.7))
                drop-shadow(0 0 40px rgba(0,245,255,0.3));
    }
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-5px); }
}
@keyframes rotateRing {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes glitch {
    0%, 95%, 100% { text-shadow: none; transform: none; }
    96%  { text-shadow: -2px 0 #ff2d55; transform: translateX(-2px); }
    97%  { text-shadow: 2px 0 #00f5ff; transform: translateX(2px); }
    98%  { text-shadow: -1px 0 #bf5fff; transform: translateX(0); }
}
"""


# ══════════════════════════════════════════════════════════════════════════
#  GSAP + Enhanced Particle System  (injected via gr.HTML)
#
#  BUG FIX vs original:
#   • `_booted` guard prevents double-init when both readyState==='complete'
#     AND the 1200ms setTimeout fallback both fire.
#   • mouseenter/mouseleave replace mouseover/mouseout to stop event
#     bubbling causing scale flicker on child elements inside buttons.
# ══════════════════════════════════════════════════════════════════════════
SCRIPTS_HTML = """
<!-- GSAP 3 CDN -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>

<!-- Top loading bar -->
<div id="hs-loader-bar"></div>

<!-- Particle canvas -->
<canvas id="hs-particles" style="
    position:fixed;top:0;left:0;
    width:100%;height:100%;
    pointer-events:none;z-index:0;
    opacity:0.8;
"></canvas>

<script>
(function () {
    /* ── BUG FIX: single-boot guard ──────────────────────────────── */
    if (window._hsBooted) return;

    function boot() {
        if (window._hsBooted) return;          // guard: second call → skip
        if (typeof gsap === 'undefined') {
            setTimeout(boot, 80);
            return;
        }
        window._hsBooted = true;

        /* ══════════════════════════════════════════════════════════
           PARTICLE SYSTEM  (cyan + purple network nodes)
        ══════════════════════════════════════════════════════════ */
        const cv  = document.getElementById('hs-particles');
        if (!cv) return;
        const ctx = cv.getContext('2d');

        function resize() {
            cv.width  = window.innerWidth;
            cv.height = window.innerHeight;
        }
        resize();
        window.addEventListener('resize', resize);

        const N = 80;
        const CONN_DIST = 120;

        const P = Array.from({ length: N }, () => ({
            x:  Math.random() * cv.width,
            y:  Math.random() * cv.height,
            vx: (Math.random() - 0.5) * 0.28,
            vy: (Math.random() - 0.5) * 0.28,
            r:  Math.random() * 1.8 + 0.5,
            a:  Math.random() * 0.35 + 0.08,
            c:  Math.random() > 0.5 ? '0,245,255' : '191,95,255',
            pulse: Math.random() * Math.PI * 2,
        }));

        /* Occasional "shooting star" streaks */
        const streaks = [];
        function spawnStreak() {
            streaks.push({
                x: Math.random() * cv.width,
                y: Math.random() * cv.height * 0.4,
                len: 60 + Math.random() * 80,
                speed: 3 + Math.random() * 4,
                angle: Math.PI / 4 + (Math.random() - 0.5) * 0.4,
                life: 1,
            });
        }
        setInterval(spawnStreak, 3500);

        let frame = 0;
        function tick() {
            ctx.clearRect(0, 0, cv.width, cv.height);
            frame++;

            /* Particles */
            P.forEach(p => {
                p.x += p.vx; p.y += p.vy;
                p.pulse += 0.02;
                if (p.x < -10) p.x = cv.width + 10;
                if (p.x > cv.width + 10) p.x = -10;
                if (p.y < -10) p.y = cv.height + 10;
                if (p.y > cv.height + 10) p.y = -10;

                const pulseR = p.r + Math.sin(p.pulse) * 0.5;
                ctx.beginPath();
                ctx.arc(p.x, p.y, pulseR, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${p.c},${p.a})`;
                ctx.fill();
            });

            /* Connections */
            for (let i = 0; i < N; i++) {
                for (let j = i + 1; j < N; j++) {
                    const dx = P[i].x - P[j].x, dy = P[i].y - P[j].y;
                    const d  = Math.sqrt(dx * dx + dy * dy);
                    if (d < CONN_DIST) {
                        const alpha = (1 - d / CONN_DIST) * 0.09;
                        ctx.beginPath();
                        ctx.moveTo(P[i].x, P[i].y);
                        ctx.lineTo(P[j].x, P[j].y);
                        ctx.strokeStyle = `rgba(0,245,255,${alpha})`;
                        ctx.lineWidth = 0.5;
                        ctx.stroke();
                    }
                }
            }

            /* Shooting streaks */
            for (let i = streaks.length - 1; i >= 0; i--) {
                const s = streaks[i];
                const ex = s.x + Math.cos(s.angle) * s.len;
                const ey = s.y + Math.sin(s.angle) * s.len;
                const grad = ctx.createLinearGradient(s.x, s.y, ex, ey);
                grad.addColorStop(0, `rgba(0,245,255,0)`);
                grad.addColorStop(0.5, `rgba(0,245,255,${s.life * 0.5})`);
                grad.addColorStop(1, `rgba(0,245,255,0)`);
                ctx.beginPath();
                ctx.moveTo(s.x, s.y);
                ctx.lineTo(ex, ey);
                ctx.strokeStyle = grad;
                ctx.lineWidth = 1.5;
                ctx.stroke();
                s.x += Math.cos(s.angle) * s.speed;
                s.y += Math.sin(s.angle) * s.speed;
                s.life -= 0.018;
                if (s.life <= 0 || s.x > cv.width + 100) streaks.splice(i, 1);
            }

            requestAnimationFrame(tick);
        }
        tick();

        /* ══════════════════════════════════════════════════════════
           PAGE-LOAD ENTRANCE  (GSAP timeline)
        ══════════════════════════════════════════════════════════ */
        const tl = gsap.timeline({ delay: 0.4 });

        const header = document.querySelector('.hs-header');
        if (header) {
            tl.from(header, { opacity: 0, y: -50, duration: 1.0, ease: 'power3.out' });
        }

        const inputSec = document.querySelector('.hs-input-section');
        if (inputSec) {
            tl.from(inputSec, { opacity: 0, y: 32, duration: 0.85, ease: 'power2.out' }, '-=0.5');
        }

        const tabs = document.querySelector('.tabs');
        if (tabs) {
            tl.from(tabs, { opacity: 0, y: 40, duration: 0.85, ease: 'power2.out' }, '-=0.45');
        }

        /* Staggered badge entrance inside header */
        const badges = document.querySelectorAll('.hs-badge');
        if (badges.length) {
            tl.from(badges, {
                opacity: 0, scale: 0.7, duration: 0.4,
                stagger: 0.08, ease: 'back.out(1.7)',
            }, '-=0.5');
        }

        /* ══════════════════════════════════════════════════════════
           LOADING BAR  (fires when predict/LIME buttons clicked)
        ══════════════════════════════════════════════════════════ */
        const loaderBar = document.getElementById('hs-loader-bar');

        function startLoader() {
            if (!loaderBar) return;
            gsap.killTweensOf(loaderBar);
            gsap.set(loaderBar, { width: '0%', opacity: 1 });
            gsap.to(loaderBar, { width: '75%', duration: 18, ease: 'power1.out' });
        }
        function finishLoader() {
            if (!loaderBar) return;
            gsap.killTweensOf(loaderBar);
            gsap.to(loaderBar, {
                width: '100%', duration: 0.3, ease: 'power2.out',
                onComplete: () => gsap.to(loaderBar, { opacity: 0, duration: 0.4, delay: 0.15 }),
            });
        }

        /* Attach loader to all Gradio trigger buttons */
        function attachLoaders() {
            document.querySelectorAll('button.primary, button.secondary').forEach(btn => {
                if (!btn.dataset.hsLoader) {
                    btn.dataset.hsLoader = '1';
                    btn.addEventListener('click', () => {
                        startLoader();
                        /* Observe DOM for result to finish bar */
                        const obs = new MutationObserver(() => {
                            finishLoader(); obs.disconnect();
                        });
                        obs.observe(document.body, { childList: true, subtree: true });
                        setTimeout(finishLoader, 90000);  // safety timeout
                    });
                }
            });
        }
        attachLoaders();
        /* Re-attach after Gradio re-renders */
        new MutationObserver(attachLoaders).observe(document.body, {
            childList: true, subtree: true,
        });

        /* ══════════════════════════════════════════════════════════
           BUTTON HOVER  (BUG FIX: mouseenter/leave, not over/out)
        ══════════════════════════════════════════════════════════ */
        document.addEventListener('mouseover', e => {
            const btn = e.target.closest('button:not([class*="tab"])');
            if (btn && !btn._hsHover) {
                btn._hsHover = true;
                gsap.to(btn, { scale: 1.03, duration: 0.15, ease: 'power1.out' });
            }
        });
        document.addEventListener('mouseout', e => {
            const btn = e.target.closest('button:not([class*="tab"])');
            if (btn) {
                btn._hsHover = false;
                gsap.to(btn, { scale: 1, duration: 0.2, ease: 'power1.inOut' });
            }
        });

        /* ══════════════════════════════════════════════════════════
           CUSTOM CYAN CURSOR DOT
        ══════════════════════════════════════════════════════════ */
        const dot   = document.createElement('div');
        const trail = document.createElement('div');

        dot.style.cssText = `
            position:fixed;width:8px;height:8px;border-radius:50%;
            pointer-events:none;z-index:99999;
            background:rgba(0,245,255,0.85);
            box-shadow:0 0 10px rgba(0,245,255,0.7),0 0 22px rgba(0,245,255,0.35);
            mix-blend-mode:screen;transform:translate(-50%,-50%);
        `;
        trail.style.cssText = `
            position:fixed;width:24px;height:24px;border-radius:50%;
            pointer-events:none;z-index:99998;
            border:1px solid rgba(0,245,255,0.35);
            mix-blend-mode:screen;transform:translate(-50%,-50%);
            transition:width 0.2s,height 0.2s,border-color 0.2s;
        `;
        document.body.appendChild(dot);
        document.body.appendChild(trail);

        let mx = 0, my = 0;
        document.addEventListener('mousemove', e => {
            mx = e.clientX; my = e.clientY;
            gsap.to(dot,   { x: mx, y: my, duration: 0.06, ease: 'none' });
            gsap.to(trail, { x: mx, y: my, duration: 0.22, ease: 'power2.out' });
        });

        /* Expand trail on hover over interactive elements */
        document.addEventListener('mouseover', e => {
            if (e.target.closest('button, a, input, textarea, select')) {
                trail.style.width  = '42px';
                trail.style.height = '42px';
                trail.style.borderColor = 'rgba(191,95,255,0.5)';
            }
        });
        document.addEventListener('mouseout', e => {
            if (e.target.closest('button, a, input, textarea, select')) {
                trail.style.width  = '24px';
                trail.style.height = '24px';
                trail.style.borderColor = 'rgba(0,245,255,0.35)';
            }
        });

        /* ══════════════════════════════════════════════════════════
           GLITCH EFFECT on title every ~8s
        ══════════════════════════════════════════════════════════ */
        function scheduleGlitch() {
            const delay = 7000 + Math.random() * 5000;
            setTimeout(() => {
                const h1 = document.querySelector('.hs-header h1');
                if (h1) {
                    h1.style.animation = 'none';
                    h1.style.animation = 'neonPulse 4s ease-in-out infinite, glitch 0.4s steps(1)';
                    setTimeout(() => {
                        if (h1) h1.style.animation = 'neonPulse 4s ease-in-out infinite';
                    }, 420);
                }
                scheduleGlitch();
            }, delay);
        }
        scheduleGlitch();

    } // end boot()

    /* ── Trigger: safe multi-path startup ─────────────────────── */
    if (document.readyState === 'complete') {
        boot();
    } else {
        window.addEventListener('load', boot);
    }
    /* Fallback for Gradio's dynamic hydration — only runs if not yet booted */
    setTimeout(() => { if (!window._hsBooted) boot(); }, 1500);

})();
</script>
"""


# ══════════════════════════════════════════════════════════════════════════
#  UI Builder
# ══════════════════════════════════════════════════════════════════════════
def build_ui() -> gr.Blocks:
    device_label = (
        "🖥 CPU"
        if not torch.cuda.is_available()
        else f"⚡ {torch.cuda.get_device_name(0)}"
    )
    thr_label = f"{_STATE['threshold']:.4f}"

    with gr.Blocks(
        title = "Hate Speech Detector — BLOOM-560m MTL v2",
        css   = CUSTOM_CSS,
        theme = gr.themes.Base(),
    ) as demo:

        # ── Inject GSAP + particle scripts ──────────────────────────
        gr.HTML(SCRIPTS_HTML)

        # ══════════════════════════════════════════════════════════════
        #  Header
        # ══════════════════════════════════════════════════════════════
        gr.HTML(f"""
        <div class="hs-header" style="
            text-align:center;
            padding:40px 24px 28px;
            position:relative;
            overflow:hidden;
        ">
            <!-- Corner brackets (animated via CSS) -->
            <div style="position:absolute;top:14px;left:28px;width:32px;height:32px;
                border-top:2px solid rgba(0,245,255,0.4);border-left:2px solid rgba(0,245,255,0.4);
                animation:float 3.5s ease-in-out infinite;"></div>
            <div style="position:absolute;top:14px;right:28px;width:32px;height:32px;
                border-top:2px solid rgba(0,245,255,0.4);border-right:2px solid rgba(0,245,255,0.4);
                animation:float 3.5s ease-in-out 0.8s infinite;"></div>
            <div style="position:absolute;bottom:8px;left:28px;width:32px;height:32px;
                border-bottom:2px solid rgba(0,245,255,0.4);border-left:2px solid rgba(0,245,255,0.4);
                animation:float 3.5s ease-in-out 1.6s infinite;"></div>
            <div style="position:absolute;bottom:8px;right:28px;width:32px;height:32px;
                border-bottom:2px solid rgba(0,245,255,0.4);border-right:2px solid rgba(0,245,255,0.4);
                animation:float 3.5s ease-in-out 2.4s infinite;"></div>

            <!-- Horizontal scan line -->
            <div style="position:absolute;left:0;right:0;height:1px;
                background:linear-gradient(90deg,transparent,rgba(0,245,255,0.5),transparent);
                animation:scanline 5s linear infinite;pointer-events:none;top:0;"></div>

            <!-- Version badge -->
            <div class="hs-badge" style="
                display:inline-flex;align-items:center;gap:6px;
                background:rgba(0,245,255,0.07);
                border:1px solid rgba(0,245,255,0.2);
                border-radius:20px;padding:4px 16px 4px 12px;
                font-family:var(--font-mono);font-size:0.68em;
                color:rgba(0,245,255,0.75);letter-spacing:2px;
                text-transform:uppercase;margin-bottom:16px;
            ">
                <span style="
                    display:inline-block;width:7px;height:7px;
                    background:var(--neon-cyan);border-radius:50%;
                    box-shadow:0 0 8px var(--neon-cyan);
                    animation:blink 2s ease-in-out infinite;
                "></span>
                THESIS DEMO · BLOOM-560M · MTL V2
            </div>

            <!-- Main title -->
            <h1 style="
                font-family:var(--font-display);
                font-size:clamp(1.8em,4vw,3em);
                font-weight:900;
                letter-spacing:4px;
                margin:0 0 10px 0;
                background:linear-gradient(135deg,#00f5ff 0%,#a0d8ff 40%,#bf5fff 100%);
                -webkit-background-clip:text;
                -webkit-text-fill-color:transparent;
                background-clip:text;
                animation:neonPulse 4s ease-in-out infinite;
                line-height:1.15;
            ">HATE SPEECH<br>DETECTOR</h1>

            <!-- Subtitle -->
            <p style="
                color:rgba(200,230,255,0.55);
                font-family:var(--font-mono);
                font-size:0.75em;letter-spacing:1.5px;
                margin:0 0 18px 0;text-transform:uppercase;
            ">Multi-Task Learning · CRF Viterbi · Label Smoothing · Temperature Scaling</p>

            <!-- System badges -->
            <div style="display:flex;justify-content:center;gap:10px;flex-wrap:wrap;">
                <span class="hs-badge" style="
                    background:rgba(0,0,0,0.45);
                    border:1px solid rgba(0,245,255,0.18);
                    border-radius:6px;padding:4px 14px;
                    font-family:var(--font-mono);font-size:0.72em;
                    color:rgba(0,245,255,0.75);
                ">{device_label}</span>
                <span class="hs-badge" style="
                    background:rgba(0,0,0,0.45);
                    border:1px solid rgba(191,95,255,0.18);
                    border-radius:6px;padding:4px 14px;
                    font-family:var(--font-mono);font-size:0.72em;
                    color:rgba(191,95,255,0.75);
                ">θ = {thr_label}</span>
                <span class="hs-badge" style="
                    background:rgba(0,0,0,0.45);
                    border:1px solid rgba(255,45,85,0.18);
                    border-radius:6px;padding:4px 14px;
                    font-family:var(--font-mono);font-size:0.72em;
                    color:rgba(255,45,85,0.75);
                ">ID · EN · Code-Mixed</span>
                <span class="hs-badge" style="
                    background:rgba(0,0,0,0.45);
                    border:1px solid rgba(0,255,136,0.18);
                    border-radius:6px;padding:4px 14px;
                    font-family:var(--font-mono);font-size:0.72em;
                    color:rgba(0,255,136,0.75);
                ">40K Sampel</span>
            </div>
        </div>
        """)

        # ══════════════════════════════════════════════════════════════
        #  Input Section
        # ══════════════════════════════════════════════════════════════
        gr.HTML("""
        <div class="hs-input-section" style="
            background:rgba(5,10,22,0.75);
            backdrop-filter:blur(24px);
            -webkit-backdrop-filter:blur(24px);
            border:1px solid rgba(0,245,255,0.12);
            border-radius:16px;
            padding:20px 20px 12px 20px;
            margin:0 0 14px 0;
            box-shadow:0 8px 40px rgba(0,0,0,0.5),inset 0 1px 0 rgba(255,255,255,0.04);
        ">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
                <div style="
                    width:3px;height:18px;border-radius:2px;
                    background:linear-gradient(180deg,var(--neon-cyan),var(--neon-purple));
                "></div>
                <span style="
                    font-family:var(--font-mono);font-size:0.7em;
                    color:rgba(0,245,255,0.65);letter-spacing:2px;text-transform:uppercase;
                ">◈ Input Teks Analisis</span>
            </div>
        </div>
        """)

        with gr.Row():
            text_input = gr.Textbox(
                label       = "",
                placeholder = (
                    "Masukkan teks berbahasa Indonesia, Inggris, atau campuran (code-mixed)...\n"
                    "Tekan Enter untuk prediksi cepat."
                ),
                lines     = 4,
                max_lines = 10,
            )

        # ── Examples ──────────────────────────────────────────────
        with gr.Accordion("◈  Contoh Teks Siap Pakai", open=False):
            gr.Examples(
                examples          = [[ex[0]] for ex in EXAMPLES],
                inputs            = [text_input],
                label             = "",
                examples_per_page = 5,
            )
            gr.HTML(
                "<div style='margin-top:8px;font-family:var(--font-mono);font-size:0.72em;"
                "color:rgba(0,245,255,0.4);line-height:1.8;'>"
                + "  ·  ".join(
                    f"<span style='color:rgba(0,245,255,0.7);'>{i+1}.</span> "
                    f"<span style='color:rgba(200,230,255,0.5);'>{ex[1]}</span>"
                    for i, ex in enumerate(EXAMPLES)
                )
                + "</div>"
            )

        # ══════════════════════════════════════════════════════════════
        #  Tabs
        # ══════════════════════════════════════════════════════════════
        with gr.Tabs():

            # ── Tab 1: Prediksi ────────────────────────────────────
            with gr.Tab("⚡ PREDIKSI REAL-TIME"):

                gr.HTML("""
                <div style="
                    display:flex;align-items:center;gap:8px;
                    padding:10px 0 4px 0;
                ">
                    <div style="width:3px;height:20px;border-radius:2px;
                        background:linear-gradient(180deg,#00f5ff,#0080ff);"></div>
                    <span style="font-family:var(--font-display);font-size:0.68em;
                        color:rgba(0,245,255,0.7);letter-spacing:2px;text-transform:uppercase;">
                        Klasifikasi + BIO Tagging (CRF Viterbi)
                    </span>
                </div>
                """)

                with gr.Row():
                    btn_predict = gr.Button(
                        "⚡  DETEKSI HATE SPEECH",
                        variant = "primary",
                        scale   = 2,
                    )

                status_bar = gr.Textbox(
                    label            = "◈  SYSTEM LOG",
                    interactive      = False,
                    lines            = 1,
                    show_copy_button = False,
                    elem_classes     = ["status-box"],
                )

                gr.HTML('<div class="glow-divider"></div>')

                with gr.Row():
                    verdict_out = gr.HTML(
                        label = "◈  Hasil Klasifikasi",
                        value = (
                            "<div style='padding:20px;text-align:center;"
                            "font-family:JetBrains Mono,monospace;font-size:0.85em;"
                            "color:rgba(0,245,255,0.3);letter-spacing:1px;'>"
                            "[ Menunggu input... ]</div>"
                        ),
                    )

                gr.HTML("""
                <div style="display:flex;align-items:center;gap:8px;padding:12px 0 4px 0;">
                    <div style="width:3px;height:18px;border-radius:2px;
                        background:linear-gradient(180deg,#ff2d55,#ff8c00);"></div>
                    <span style="font-family:var(--font-display);font-size:0.65em;
                        color:rgba(255,45,85,0.65);letter-spacing:2px;text-transform:uppercase;">
                        Token-Level BIO Tagging
                    </span>
                </div>
                """)

                highlight_out = gr.HTML(
                    label = "Highlighted Text",
                    value = (
                        "<div style='padding:16px;font-family:JetBrains Mono,monospace;"
                        "font-size:0.85em;color:rgba(0,245,255,0.3);'>[ — ]</div>"
                    ),
                )

                gr.HTML("<br>")
                bio_table_out = gr.Dataframe(
                    headers       = ["No", "Token", "BIO Label", "Keterangan"],
                    label         = "◈  BIO TABLE",
                    # BUG FIX: was ["number","str","str","str"]; build_bio_table
                    # returns str(i) for No → use "str" to avoid type mismatch
                    datatype      = ["str", "str", "str", "str"],
                    interactive   = False,
                    wrap          = True,
                    column_widths = ["6%", "30%", "15%", "49%"],
                )

                btn_predict.click(
                    fn      = fn_predict,
                    inputs  = [text_input],
                    outputs = [verdict_out, highlight_out, bio_table_out, status_bar],
                )
                text_input.submit(
                    fn      = fn_predict,
                    inputs  = [text_input],
                    outputs = [verdict_out, highlight_out, bio_table_out, status_bar],
                )

                gr.HTML("""
                <details>
                  <summary>Cara membaca output ini</summary>
                  <ul style="margin-top:10px;list-style:none;padding:0;
                      font-family:var(--font-mono);font-size:0.8em;
                      color:rgba(200,230,255,0.55);line-height:2;">
                    <li><span style="color:var(--neon-cyan);">P(hate)</span>
                        — Probabilitas kalimat mengandung hate speech (0–1)</li>
                    <li><span style="color:var(--neon-cyan);">Threshold (θ)</span>
                        — Batas klasifikasi dari optimasi validasi set NB05</li>
                    <li><span style="color:#ff4444;font-weight:bold;">B-TOX</span>
                        — Awal span toxic (Beginning)</li>
                    <li><span style="color:#ff8c00;">I-TOX</span>
                        — Lanjutan span toxic (Inside)</li>
                    <li><span style="color:var(--neon-green);">O</span>
                        — Token tidak toxic (Outside)</li>
                  </ul>
                </details>
                """)

            # ── Tab 2: LIME ────────────────────────────────────────
            with gr.Tab("🔬 ANALISIS LIME"):

                gr.HTML("""
                <div style="
                    background:rgba(191,95,255,0.05);
                    border:1px solid rgba(191,95,255,0.18);
                    border-radius:10px;
                    padding:14px 18px;margin-bottom:14px;
                    font-family:var(--font-body);font-size:0.87em;
                    color:rgba(200,220,255,0.65);line-height:1.6;
                ">
                    <span style="color:rgba(191,95,255,0.9);font-weight:700;
                        font-family:var(--font-display);font-size:0.9em;letter-spacing:1px;">
                        LIME
                    </span>
                    &nbsp;<em>(Local Interpretable Model-agnostic Explanations)</em>
                    — menjelaskan kontribusi tiap kata terhadap prediksi model.
                    Pastikan teks sudah diisi, lalu klik tombol di bawah.
                </div>
                """)

                with gr.Row():
                    n_samples_slider = gr.Slider(
                        minimum = 50,
                        maximum = 500,
                        value   = DEFAULT_LIME_SAMPLES,
                        step    = 50,
                        label   = "◈  JUMLAH SAMPEL LIME",
                        scale   = 3,
                    )
                    btn_lime = gr.Button(
                        "🔬  JALANKAN LIME",
                        variant = "secondary",
                        scale   = 1,
                    )

                gr.HTML("""
                <div style="
                    background:rgba(255,170,0,0.06);
                    border:1px solid rgba(255,170,0,0.2);
                    border-radius:8px;padding:10px 16px;
                    font-family:var(--font-mono);font-size:0.78em;
                    color:rgba(255,190,80,0.75);margin-bottom:12px;
                    line-height:1.7;
                ">
                    ⏳&nbsp; <b>ESTIMASI WAKTU</b> (200 sampel):
                    CPU ~30–120 detik &nbsp;·&nbsp; GPU ~5–15 detik<br>
                    <span style="color:rgba(255,190,80,0.45);">
                        ↳ Jangan tutup browser selama proses berlangsung
                    </span>
                </div>
                """)

                lime_img_out = gr.Image(
                    label  = "◈  LIME WORD CONTRIBUTIONS",
                    type   = "pil",
                    height = 520,
                )
                lime_summary_out = gr.Textbox(
                    label        = "◈  RINGKASAN LIME",
                    interactive  = False,
                    lines        = 3,
                    elem_classes = ["status-box"],
                )

                btn_lime.click(
                    fn      = fn_lime,
                    inputs  = [text_input, n_samples_slider],
                    outputs = [lime_img_out, lime_summary_out],
                )

                gr.HTML("""
                <details>
                  <summary>Keterbatasan LIME (NB05v2 Bab 4)</summary>
                  <ol style="margin-top:10px;padding-left:18px;
                      font-family:var(--font-mono);font-size:0.8em;
                      color:rgba(200,220,255,0.5);line-height:2;">
                    <li>BLOOM menggunakan
                        <b style="color:var(--neon-purple);">BPE tokenizer</b>;
                        LIME beroperasi di
                        <b style="color:var(--neon-purple);">level kata</b>.
                        Mismatch subword → kontribusi bersifat approksimasi.</li>
                    <li>LIME adalah pendekatan
                        <b style="color:var(--neon-purple);">lokal-linear</b>
                        → hanya valid di sekitar prediksi spesifik ini.</li>
                    <li>Gunakan sebagai analisis
                        <b style="color:var(--neon-purple);">KUALITATIF</b>,
                        bukan kuantitatif.</li>
                  </ol>
                </details>
                """)

        # ── Footer ──────────────────────────────────────────────────
        gr.HTML("""
        <div style="
            text-align:center;
            border-top:1px solid rgba(0,245,255,0.08);
            padding:16px 0 8px 0;
            margin-top:24px;
        ">
            <div style="
                font-family:var(--font-mono);font-size:0.68em;
                color:rgba(200,230,255,0.2);letter-spacing:0.5px;
                line-height:1.9;
            ">
                BLOOM-560m MTL v2 &nbsp;·&nbsp; CRF + Label Smoothing
                &nbsp;·&nbsp; Skripsi Demo
                &nbsp;·&nbsp; 40,000 Sampel (ID + EN + Code-Mixed)
            </div>
            <div style="
                font-family:var(--font-mono);font-size:0.62em;
                color:rgba(0,245,255,0.15);margin-top:4px;
            ">
                ⬡ Glassmorphism UI · GSAP 3 Particle System · Orbitron · JetBrains Mono
            </div>
        </div>
        """)

    return demo


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Hate Speech Demo — BLOOM-560m MTL v2"
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Path ke best_model.pt")
    parser.add_argument("--bloom",      default=DEFAULT_BLOOM_PATH,
                        help="Path ke folder bloom-560m-local")
    parser.add_argument("--threshold",  default=DEFAULT_THRESHOLD_JSON,
                        help="Path ke threshold_info_v2.json")
    parser.add_argument("--lime-samples", type=int, default=DEFAULT_LIME_SAMPLES,
                        help=f"Jumlah sampel LIME (default: {DEFAULT_LIME_SAMPLES})")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port server lokal (default: 7860)")
    args = parser.parse_args()

    missing = []
    if not os.path.exists(args.checkpoint):
        missing.append(f"  ❌ Checkpoint : {args.checkpoint}")
    if not os.path.isdir(args.bloom):
        missing.append(f"  ❌ BLOOM lokal : {args.bloom}")
    if missing:
        print("\n" + "="*60)
        print(" GAGAL: File model tidak ditemukan!")
        print("="*60)
        for m in missing: print(m)
        print("\nIkuti langkah di DEMO_GUIDE.md untuk mengunduh model.")
        print("="*60)
        sys.exit(1)

    print("\n" + "="*60)
    print(" Hate Speech Demo — BLOOM-560m MTL v2 [Futuristic UI]")
    print("="*60)
    _initialize(
        checkpoint     = args.checkpoint,
        bloom_path     = args.bloom,
        threshold_json = args.threshold,
        lime_samples   = args.lime_samples,
    )

    ui = build_ui()
    ui.launch(
        server_name = "0.0.0.0",
        server_port = args.port,
        share       = False,
        inbrowser   = True,
        show_error  = True,
    )


if __name__ == "__main__":
    main()
