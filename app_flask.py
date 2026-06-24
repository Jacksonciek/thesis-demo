import argparse
import base64
import io
import json
import os
import sys

import torch
from flask import Flask, jsonify, render_template_string, request

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

from inference import predict_text, build_highlighted_html, build_bio_table
from lime_viz  import run_lime_explanation, fig_to_pil
from model_utils import load_model, load_tokenizer, load_threshold

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT     = os.path.join(_DEMO_DIR, "models", "main_bloom_mtl_v2", "best_model_hm.pt")
DEFAULT_BLOOM_PATH     = os.path.join(_DEMO_DIR, "models", "bloom-560m-local")
DEFAULT_THRESHOLD_JSON = os.path.join(_DEMO_DIR, "models", "threshold_info_v2.json")
DEFAULT_LIME_SAMPLES   = 200

_STATE = {
    "model":     None,
    "tokenizer": None,
    "threshold": 0.4429,
    "device":    None,
    "ready":     False,
}

app = Flask(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  Initialization
# ══════════════════════════════════════════════════════════════════════════
def _initialize(checkpoint, bloom_path, threshold_json):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _STATE["device"] = device
    print(f"[App] Device: {device}")
    _STATE["tokenizer"] = load_tokenizer(bloom_path)
    _STATE["model"]     = load_model(checkpoint, bloom_path, device)
    _STATE["threshold"] = load_threshold(threshold_json)
    _STATE["ready"]     = True
    print("[App] ✅ Model siap.")


# ══════════════════════════════════════════════════════════════════════════
#  Routes
# ══════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    device_label = (
        "CPU"
        if not torch.cuda.is_available()
        else torch.cuda.get_device_name(0)
    )
    return render_template_string(
        HTML_TEMPLATE,
        device_label=device_label,
        threshold=f"{_STATE['threshold']:.4f}",
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not _STATE["ready"]:
        return jsonify({"error": "Model belum siap"}), 503

    body = request.get_json(force=True)
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Teks kosong"}), 400

    result = predict_text(
        text      = text,
        tokenizer = _STATE["tokenizer"],
        model     = _STATE["model"],
        device    = _STATE["device"],
        threshold = _STATE["threshold"],
    )

    highlight_html = build_highlighted_html(result["words"], result["word_bio"])
    bio_rows       = build_bio_table(result["words"], result["word_bio"])

    return jsonify({
        "pred":         result["pred"],
        "prob_hate":    result["prob_hate"],
        "prob_nonhate": result["prob_nonhate"],
        "conf":         result["conf"],
        "threshold":    _STATE["threshold"],
        "spans":        result["spans"],
        "highlight_html": highlight_html,
        "bio_rows":     bio_rows,
    })


@app.route("/api/lime", methods=["POST"])
def api_lime():
    if not _STATE["ready"]:
        return jsonify({"error": "Model belum siap"}), 503

    body      = request.get_json(force=True)
    text      = (body.get("text") or "").strip()
    # Cap diseragamkan dengan slider HTML (max="500") agar konsisten
    n_samples = max(50, min(int(body.get("n_samples", DEFAULT_LIME_SAMPLES)), 500))

    if not text:
        return jsonify({"error": "Teks kosong"}), 400

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
        threshold    = _STATE["threshold"],  # BUG FIX: teruskan threshold 0.4429
    )
    if fig is None:
        return jsonify({"error": "LIME gagal — teks terlalu pendek?"}), 400

    pil_img = fig_to_pil(fig)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    top3 = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:3]

    return jsonify({
        "image_b64":   img_b64,
        "pred_label":  pred_label,
        "confidence":  confidence,
        "top3":        top3,
        "n_samples":   n_samples,
    })


# ══════════════════════════════════════════════════════════════════════════
#  HTML Template
# ══════════════════════════════════════════════════════════════════════════
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Hate Speech Detector</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Oxanium:wght@300;400;600;700;800&family=Exo+2:ital,wght@0,300;0,400;0,500;0,600;1,300&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#060c1e;--glass:rgba(255,255,255,0.048);--glass2:rgba(255,255,255,0.085);
  --border:rgba(255,255,255,0.07);--border2:rgba(180,160,255,0.18);
  --v:#7c3aed;--v2:#a78bfa;--v3:#ede9fe;--c:#06b6d4;--c2:#67e8f9;
  --h:#f43f5e;--g:#10b981;--a:#f59e0b;
  --t1:#f0f4ff;--t2:rgba(210,220,255,0.72);--t3:rgba(180,200,255,0.42);--t4:rgba(160,185,255,0.22);
  --fh:'Oxanium',sans-serif;--fb:'Exo 2',sans-serif;--fm:'JetBrains Mono',monospace;
  --r:10px;--r2:16px;
}
html,body{height:100%;overflow:hidden;background:var(--bg);color:var(--t1);-webkit-font-smoothing:antialiased;font-family:var(--fb)}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-thumb{background:rgba(124,58,237,.3);border-radius:2px}

/* ─── SPLASH ─── */
#splash{
  position:fixed;inset:0;z-index:900;
  display:flex;align-items:center;justify-content:center;
  background:var(--bg);
  clip-path:circle(150% at 50% 50%);
  transition:clip-path 1.8s cubic-bezier(.86,0,.07,1),
             opacity 0.7s ease 1.1s,
             filter 0.9s ease 0.15s;
  filter:blur(0px);
  will-change:clip-path,filter;
}
#splash.vanish{
  clip-path:circle(0% at 50% 50%);
  opacity:0;
  filter:blur(18px) brightness(1.3);
  pointer-events:none;
}
#pCanvas{position:absolute;inset:0;width:100%;height:100%}
.sp-orb{position:absolute;border-radius:50%;pointer-events:none}
.sp-orb1{width:85vw;height:85vw;max-width:900px;max-height:900px;background:radial-gradient(ellipse at 40% 40%,rgba(124,58,237,.55) 0%,rgba(79,30,190,.22) 35%,transparent 70%);top:-30%;left:-20%;filter:blur(60px);animation:oF1 14s ease-in-out infinite alternate}
.sp-orb2{width:65vw;height:65vw;max-width:700px;max-height:700px;background:radial-gradient(ellipse,rgba(6,182,212,.3) 0%,transparent 65%);bottom:-20%;right:-15%;filter:blur(70px);animation:oF2 18s ease-in-out infinite alternate}
.sp-orb3{width:40vw;height:40vw;max-width:420px;max-height:420px;background:radial-gradient(ellipse,rgba(192,38,211,.18) 0%,transparent 65%);top:40%;left:55%;filter:blur(80px);animation:oF1 22s ease-in-out infinite alternate-reverse}
@keyframes oF1{from{transform:translate(0,0) scale(1)}to{transform:translate(3%,-5%) scale(1.07)}}
@keyframes oF2{from{transform:translate(0,0) scale(1)}to{transform:translate(-4%,3%) scale(1.06)}}
.rings{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;pointer-events:none}
.ring{position:absolute;border-radius:50%;border:1px solid}
.ring1{width:480px;height:480px;border-color:rgba(124,58,237,.12);animation:spin 30s linear infinite}
.ring2{width:680px;height:680px;border-color:rgba(6,182,212,.08);animation:spin 45s linear infinite reverse}
.ring3{width:880px;height:880px;border-color:rgba(124,58,237,.05);animation:spin 60s linear infinite}
.ring4{width:200px;height:200px;border-color:rgba(124,58,237,.15);animation:spin 20s linear infinite reverse}
@keyframes spin{to{transform:rotate(360deg)}}
.ring1::before{content:'';position:absolute;width:7px;height:7px;border-radius:50%;background:#a78bfa;top:-3.5px;left:50%;transform:translateX(-50%);box-shadow:0 0 12px #a78bfa,0 0 24px rgba(167,139,250,.6)}
.ring2::before{content:'';position:absolute;width:6px;height:6px;border-radius:50%;background:#67e8f9;top:-3px;left:50%;transform:translateX(-50%);box-shadow:0 0 10px #67e8f9,0 0 20px rgba(103,232,249,.5)}
#splash::after{content:'';position:absolute;inset:0;pointer-events:none;z-index:1;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.03'/%3E%3C/svg%3E");background-size:200px 200px}
.sp-inner{position:relative;z-index:2;text-align:center;display:flex;flex-direction:column;align-items:center}
.sp-eyebrow{font-family:var(--fm);font-size:10px;letter-spacing:4px;text-transform:uppercase;color:rgba(167,139,250,.8);background:rgba(124,58,237,.1);border:1px solid rgba(124,58,237,.22);border-radius:100px;padding:7px 22px;margin-bottom:40px;animation:sUp .9s cubic-bezier(.16,1,.3,1) .3s both}
.sp-title{font-family:var(--fh);font-size:clamp(52px,8.5vw,108px);font-weight:800;letter-spacing:-3px;line-height:.92;margin-bottom:28px;background:linear-gradient(145deg,#ffffff 0%,#c4b5fd 22%,var(--c2) 52%,#a5f3fc 82%,#f0f4ff 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:sUp 1s cubic-bezier(.16,1,.3,1) .5s both;position:relative;text-align:center}
.sp-title::after{content:'';position:absolute;bottom:-6px;left:8%;right:8%;height:1px;background:linear-gradient(90deg,transparent,rgba(167,139,250,.5),rgba(103,232,249,.5),transparent)}
.sp-sub{font-family:var(--fb);font-size:clamp(12px,1.4vw,15px);font-weight:300;color:var(--t3);letter-spacing:.03em;line-height:1.9;max-width:540px;margin-bottom:12px;animation:sUp .95s cubic-bezier(.16,1,.3,1) .7s both}
.sp-thesis{font-family:var(--fm);font-size:8.5px;letter-spacing:2px;text-transform:uppercase;color:var(--t4);margin-bottom:70px;animation:sUp .9s cubic-bezier(.16,1,.3,1) .85s both}
.sp-press{display:flex;align-items:center;gap:18px;animation:sUp .9s cubic-bezier(.16,1,.3,1) 1.1s both;cursor:pointer}
.sp-press span{font-family:var(--fm);font-size:10px;letter-spacing:5px;text-transform:uppercase;color:var(--t3);animation:pBreath 3s ease-in-out 2s infinite;white-space:nowrap}
.press-line{width:65px;height:1px;background:linear-gradient(90deg,transparent,rgba(167,139,250,.4));animation:pBreath 3s ease-in-out 2s infinite}
.press-line:last-child{transform:scaleX(-1)}
.sp-press:hover span,.sp-press:hover .press-line{color:var(--v2);opacity:1}
@keyframes pBreath{0%,100%{opacity:.3}50%{opacity:1}}
@keyframes sUp{from{opacity:0;transform:translateY(28px)}to{opacity:1;transform:none}}
.sp-bracket{position:absolute;width:28px;height:28px;pointer-events:none;animation:sUp .9s cubic-bezier(.16,1,.3,1) 1.4s both}
.sp-bracket.tl{top:-55px;left:-70px;border-top:1.5px solid rgba(124,58,237,.4);border-left:1.5px solid rgba(124,58,237,.4)}
.sp-bracket.tr{top:-55px;right:-70px;border-top:1.5px solid rgba(6,182,212,.4);border-right:1.5px solid rgba(6,182,212,.4)}
.sp-bracket.bl{bottom:-55px;left:-70px;border-bottom:1.5px solid rgba(124,58,237,.4);border-left:1.5px solid rgba(124,58,237,.4)}
.sp-bracket.br{bottom:-55px;right:-70px;border-bottom:1.5px solid rgba(6,182,212,.4);border-right:1.5px solid rgba(6,182,212,.4)}

/* ─── TRANSITION ─── */
#transSeq{
  position:fixed;inset:0;z-index:800;pointer-events:none;
  display:flex;align-items:center;justify-content:center;
  opacity:0;
  transition:opacity .25s ease;
  backdrop-filter:blur(0px);
  -webkit-backdrop-filter:blur(0px);
}
#transSeq.active{opacity:1;backdrop-filter:blur(2px);-webkit-backdrop-filter:blur(2px);transition:opacity .18s ease,backdrop-filter .4s ease}
#transSeq.done{opacity:0;backdrop-filter:blur(0px);-webkit-backdrop-filter:blur(0px);transition:opacity .9s ease .05s,backdrop-filter .9s ease .05s}
.scan-h{position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent 0%,rgba(103,232,249,.9) 20%,rgba(167,139,250,1) 50%,rgba(103,232,249,.9) 80%,transparent 100%);box-shadow:0 0 20px rgba(167,139,250,1),0 0 50px rgba(103,232,249,.5);animation:scanDn .8s cubic-bezier(.4,0,.6,1) forwards}
@keyframes scanDn{0%{top:0;opacity:0}5%{opacity:1}100%{top:100vh;opacity:.05}}
.init-txt{font-family:var(--fh);font-size:clamp(18px,3vw,32px);font-weight:700;letter-spacing:12px;text-transform:uppercase;background:linear-gradient(90deg,var(--v2),var(--c2),var(--v2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;opacity:0;transition:opacity .7s ease;position:absolute}
.init-txt.visible{opacity:1;animation:initG .75s ease both}
@keyframes initG{0%{letter-spacing:38px;filter:blur(14px)}45%{letter-spacing:9px;filter:blur(0)}68%{letter-spacing:15px}83%{letter-spacing:11px}100%{letter-spacing:12px}}

/* ─── APP BACKGROUND ─── */
.bg-wrap{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden}
.orb{position:absolute;border-radius:50%}
.o1{width:80vw;height:80vw;max-width:920px;max-height:920px;background:radial-gradient(ellipse at 35% 40%,rgba(109,40,217,.52) 0%,rgba(79,30,190,.22) 30%,transparent 68%);top:-25%;left:-20%;filter:blur(70px);animation:oF1 20s ease-in-out infinite alternate}
.o2{width:60vw;height:60vw;max-width:700px;max-height:700px;background:radial-gradient(ellipse,rgba(6,182,212,.3) 0%,transparent 65%);bottom:-20%;right:-10%;filter:blur(80px);animation:oF2 25s ease-in-out infinite alternate}
.o3{width:45vw;height:45vw;max-width:520px;background:radial-gradient(ellipse,rgba(192,38,211,.2) 0%,transparent 65%);top:30%;right:25%;filter:blur(90px);animation:oF1 18s ease-in-out infinite alternate-reverse}
.o4{width:35vw;height:35vw;max-width:380px;background:radial-gradient(ellipse,rgba(16,185,129,.14) 0%,transparent 65%);bottom:10%;left:20%;filter:blur(100px);animation:oF2 30s ease-in-out infinite alternate}
.grid-overlay{position:absolute;inset:0;background-image:linear-gradient(rgba(124,58,237,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(124,58,237,.03) 1px,transparent 1px);background-size:52px 52px;mask-image:radial-gradient(ellipse 80% 80% at 50% 50%,black 40%,transparent 100%)}

/* ─── APP SHELL ─── */
#app{
  position:relative;z-index:1;height:100vh;display:flex;flex-direction:column;
  opacity:0;pointer-events:none;
  transform:translateY(22px) scale(0.992);
  transition:opacity 1.2s cubic-bezier(.16,1,.3,1),
             transform 1.2s cubic-bezier(.16,1,.3,1);
  will-change:opacity,transform;
}
#app.show{opacity:1;pointer-events:all;transform:none}
@keyframes blink{0%,100%{opacity:.3}50%{opacity:1}}

/* ─── WORKSPACE ─── */
/* Default: centered input stage */
.ws{
  flex:1 1 0;min-height:0;
  display:flex;align-items:center;justify-content:center;
  position:relative;z-index:1;
}

/* LEFT PANEL — centered by default, becomes side panel after split */
.lp{
  display:flex;flex-direction:column;gap:16px;
  width:min(640px,calc(100% - 48px));
  padding:0;
  flex-shrink:0;
  transition:width 1s cubic-bezier(.16,1,.3,1),
             padding 1s cubic-bezier(.16,1,.3,1),
             border-right-color .6s ease;
  border-right:1px solid transparent;
  overflow-y:auto;
  max-height:100vh;
}

/* RIGHT PANEL — hidden until split */
.rp{
  flex:0 0 0;
  width:0;
  overflow:hidden;
  opacity:0;
  transition:flex 1s cubic-bezier(.16,1,.3,1),
             opacity .7s ease .45s;
}
.rp-inner{
  width:min(720px,90vw);padding:24px 26px;min-height:100%;
  display:flex;flex-direction:column;
}

/* SPLIT STATE */
#app.split .ws{
  align-items:stretch;
  justify-content:flex-start;
}
#app.split .lp{
  width:380px;
  padding:22px 22px 18px;
  border-right-color:var(--border);
}
#app.split .rp{
  flex:1 1 0;
  width:auto;
  overflow-y:auto;
  opacity:1;
  position:relative;
}
#app.split .rp-inner{
  width:100%;
}

/* ─── LOADING OVERLAY (sits on top of rp, never touches rpInner DOM) ─── */
.rp-overlay{
  position:absolute;inset:0;z-index:20;
  display:none;align-items:center;justify-content:center;
  background:rgba(6,10,24,.55);
  backdrop-filter:blur(6px);
  -webkit-backdrop-filter:blur(6px);
  pointer-events:all;
  opacity:0;
  transition:opacity .25s ease;
}
.rp-overlay.visible{opacity:1}
.rp-overlay-inner{
  display:flex;flex-direction:column;align-items:center;gap:16px;color:var(--t3);
}
.rp-overlay-inner span{
  font-family:var(--fm);font-size:10px;letter-spacing:4px;text-transform:uppercase;
  animation:pBreath 1.4s ease-in-out infinite;
}

/* Center-mode exclusive elements */
.cs-brand{
  display:flex;flex-direction:column;align-items:center;gap:10px;
  transition:opacity .4s,max-height .5s,margin .5s;
  overflow:hidden;
  max-height:200px;
  opacity:1;
}
.cs-title{
  font-family:var(--fh);font-size:clamp(36px,5vw,58px);font-weight:800;letter-spacing:1px;
  background:linear-gradient(135deg,#fff 0%,#c4b5fd 30%,var(--c2) 65%,#f0f4ff 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.cs-title em{font-style:normal}
.cs-eyebrow{font-family:var(--fm);font-size:10px;letter-spacing:4px;text-transform:uppercase;color:rgba(167,139,250,.7);background:rgba(124,58,237,.08);border:1px solid rgba(124,58,237,.18);border-radius:100px;padding:5px 18px}
.cs-sub{font-family:var(--fb);font-size:13px;font-weight:300;color:var(--t3);letter-spacing:.02em;text-align:center;line-height:1.7}

/* Split-mode brand (small header in lp) */
.sp-brand{
  display:none;align-items:center;gap:10px;
  font-family:var(--fh);font-size:17px;font-weight:700;
  padding-bottom:14px;border-bottom:1px solid var(--border);
  flex-shrink:0;
}
.sp-brand em{font-style:normal;color:var(--v2)}
.sp-brand-sub{font-family:var(--fm);font-size:8px;letter-spacing:3px;text-transform:uppercase;color:var(--t4);margin-left:auto}
.sp-brand-dot{width:6px;height:6px;border-radius:50%;background:var(--v2);box-shadow:0 0 8px var(--v2)}

#app.split .cs-brand{opacity:0;max-height:0;margin-bottom:-16px;pointer-events:none}
#app.split .sp-brand{display:flex}

/* Textarea */
.ta-wrap{
  position:relative;
  background:rgba(8,14,36,.7);border:1px solid var(--border);
  border-radius:var(--r2);overflow:hidden;transition:border-color .2s,box-shadow .2s;
}
.ta-wrap:focus-within{border-color:rgba(124,58,237,.5);box-shadow:0 0 0 3px rgba(124,58,237,.08),0 8px 40px rgba(124,58,237,.12)}
.ta-scanline{position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(124,58,237,.7),transparent);animation:taScan 4s ease-in-out 2s infinite;opacity:0;pointer-events:none}
@keyframes taScan{0%,100%{opacity:0;top:0}10%{opacity:.8}90%{opacity:.2;top:100%}}
textarea#inputText{
  width:100%;height:130px;background:transparent;border:none;outline:none;resize:none;
  color:var(--t1);font-family:var(--fb);font-size:15px;line-height:1.85;
  padding:18px 20px;caret-color:var(--v2);
}
textarea#inputText::placeholder{color:var(--t4);font-style:italic;font-size:14px}
#app.split textarea#inputText{height:140px}

/* Buttons */
.btn-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.char-ct{font-family:var(--fm);font-size:10px;color:var(--t4);margin-left:auto}
.btn-p{
  display:inline-flex;align-items:center;gap:8px;padding:11px 28px;
  background:rgba(124,58,237,.12);border:1px solid rgba(124,58,237,.35);
  border-radius:10px;color:var(--v2);font-family:var(--fh);font-size:12px;font-weight:600;
  letter-spacing:2px;text-transform:uppercase;cursor:pointer;
  position:relative;overflow:hidden;transition:color .18s,box-shadow .18s,border-color .18s;
}
.btn-p::before{content:'';position:absolute;inset:0;background:linear-gradient(115deg,rgba(109,40,217,.9),rgba(6,182,212,.75));transform:translateX(-102%);transition:transform .3s cubic-bezier(.4,0,.2,1)}
.btn-p:hover::before{transform:none}
.btn-p:hover{color:#fff;border-color:transparent;box-shadow:0 4px 32px rgba(109,40,217,.5)}
.btn-p:disabled{opacity:.3;cursor:not-allowed}
.btn-p>*{position:relative;z-index:1}
.btn-s{
  display:inline-flex;align-items:center;gap:6px;padding:11px 20px;
  background:rgba(255,255,255,.03);border:1px solid var(--border);
  border-radius:10px;color:var(--t3);font-family:var(--fm);font-size:10px;letter-spacing:2px;
  text-transform:uppercase;cursor:pointer;transition:color .18s,border-color .18s,background .18s;
}
.btn-s:hover{color:var(--t2);border-color:var(--border2);background:var(--glass)}
.btn-s:disabled{opacity:.3;cursor:not-allowed}

/* Status bar */
.sbar{display:flex;align-items:center;gap:9px;font-family:var(--fm);font-size:11px;padding:10px 14px;background:rgba(6,10,24,.5);border-radius:10px;border:1px solid transparent;color:var(--t4);transition:color .25s,border-color .25s}
.sb-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0;transition:background .25s}
.s-idle .sb-dot{background:var(--t4)}
.s-load{color:var(--v2);border-color:rgba(124,58,237,.2)}.s-load .sb-dot{background:var(--v2);animation:blink .5s infinite}
.s-ok{color:var(--g);border-color:rgba(16,185,129,.2)}.s-ok .sb-dot{background:var(--g)}
.s-err{color:var(--h);border-color:rgba(244,63,94,.2)}.s-err .sb-dot{background:var(--h)}

/* LIME section */
.lime-sect{display:flex;flex-direction:column;gap:9px;flex-shrink:0}
.lime-ctrl{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.panel-lbl{display:flex;align-items:center;gap:8px;font-family:var(--fm);font-size:10px;letter-spacing:3px;text-transform:uppercase;color:var(--t4)}
.panel-lbl-dot{width:5px;height:5px;border-radius:50%;background:var(--v2);box-shadow:0 0 6px var(--v2);flex-shrink:0}
.s-lbl{font-family:var(--fm);font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--t4);white-space:nowrap}
input[type=range]{flex:1;-webkit-appearance:none;height:3px;border-radius:2px;background:linear-gradient(90deg,var(--v) var(--p,0%),rgba(255,255,255,.07) var(--p,0%));outline:none;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:var(--v2);box-shadow:0 0 10px rgba(124,58,237,.6);cursor:pointer;transition:transform .15s}
input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.3)}
.s-val{font-family:var(--fm);font-size:10px;color:var(--v2);min-width:30px;text-align:right}
.lime-cav{font-family:var(--fm);font-size:9px;color:rgba(245,158,11,.45);background:rgba(245,158,11,.04);border:1px solid rgba(245,158,11,.1);border-radius:8px;padding:7px 12px;line-height:1.65}

/* Hide LIME in center mode */
.lime-sect{
  max-height:0;overflow:hidden;opacity:0;
  transition:max-height .5s .3s,opacity .4s .35s;
  pointer-events:none;
}
#app.split .lime-sect{
  max-height:300px;opacity:1;pointer-events:all;
}

/* ─── RIGHT PANEL RESULTS ─── */
.sec-lbl-row{display:flex;align-items:center;gap:8px;font-family:var(--fm);font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--t4);margin:18px 0 9px;padding-top:6px;border-top:1px solid var(--border)}

/* Glass card */
.gc{background:var(--glass);border:1px solid var(--border);border-radius:var(--r2);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);transition:border-color .2s;margin-bottom:14px}
.gc:hover{border-color:var(--border2)}

/* Verdict */
.verdict{padding:20px 22px 18px;position:relative;overflow:hidden;transition:border-color .5s,box-shadow .5s}
.verdict.is-hate{border-color:rgba(244,63,94,.3);box-shadow:0 0 50px rgba(244,63,94,.07),inset 0 0 50px rgba(244,63,94,.03)}
.verdict.is-safe{border-color:rgba(16,185,129,.28);box-shadow:0 0 50px rgba(16,185,129,.06),inset 0 0 50px rgba(16,185,129,.02)}
.verdict.is-neutral{border-color:rgba(160,185,255,.1);box-shadow:none}
.vring{position:absolute;right:-40px;top:50%;width:200px;height:200px;border-radius:50%;border:1px solid currentColor;pointer-events:none;opacity:0;transform:translateY(-50%) scale(.12)}
.vring.r1{animation:vRA 3.2s ease-out infinite 0s}.vring.r2{animation:vRA 3.2s ease-out infinite 1.1s}.vring.r3{animation:vRA 3.2s ease-out infinite 2.2s}
@keyframes vRA{0%{transform:translateY(-50%) scale(.12);opacity:.6}100%{transform:translateY(-50%) scale(2.4);opacity:0}}
.rh{color:var(--h)}.rs{color:var(--g)}.rn{color:rgba(160,185,255,.15)}
.v-layout{display:flex;align-items:center;gap:16px;position:relative;z-index:2}
.v-icon{width:52px;height:52px;flex-shrink:0}
.v-text{flex:1;min-width:0}
.v-label{font-family:var(--fm);font-size:8px;letter-spacing:3px;text-transform:uppercase;color:var(--t4);margin-bottom:3px}
.v-big{font-family:var(--fh);font-size:clamp(30px,4vw,50px);font-weight:800;letter-spacing:3px;text-transform:uppercase;line-height:1;transition:color .5s,text-shadow .5s}
.v-big.hate{color:var(--h);text-shadow:0 0 24px rgba(244,63,94,.55),0 0 60px rgba(244,63,94,.2);animation:glitch 7s infinite}
.v-big.safe{color:var(--g);text-shadow:0 0 22px rgba(16,185,129,.45),0 0 55px rgba(16,185,129,.15)}
.v-big.neutral{color:rgba(160,185,255,.25);text-shadow:none;animation:none}
@keyframes glitch{0%,88%,100%{transform:none;clip-path:none}89%{transform:translateX(2px);clip-path:polygon(0 15%,100% 15%,100% 40%,0 40%)}90%{transform:translateX(-2px);clip-path:polygon(0 55%,100% 55%,100% 75%,0 75%)}91%{transform:none;clip-path:none}92%{transform:translateX(3px);clip-path:polygon(0 5%,100% 5%,100% 22%,0 22%)}93%{transform:none;clip-path:none}}
.v-sub{font-size:12px;font-weight:300;color:var(--t2);margin-top:5px;line-height:1.6}
.conf-wrap{flex-shrink:0;position:relative;width:76px;height:76px}
.conf-svg{width:76px;height:76px;transform:rotate(-90deg)}
.conf-bg{fill:none;stroke:rgba(255,255,255,.05);stroke-width:3.5}
.conf-fg{fill:none;stroke-width:3.5;stroke-linecap:round;stroke-dasharray:207;stroke-dashoffset:207;transition:stroke-dashoffset 1.1s cubic-bezier(.4,0,.2,1),stroke .5s}
.conf-center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.conf-num{font-family:var(--fh);font-size:16px;font-weight:700;line-height:1;transition:color .5s}
.conf-lbl-sm{font-family:var(--fm);font-size:8px;letter-spacing:1px;text-transform:uppercase;color:var(--t4);margin-top:2px}

/* Prob bars */
.prob-row{display:flex;align-items:center;gap:12px;margin-bottom:12px}
.prob-lbl{font-family:var(--fm);font-size:10px;letter-spacing:2px;text-transform:uppercase;width:50px;flex-shrink:0}
.prob-bar-bg{flex:1;height:6px;background:rgba(255,255,255,.06);border-radius:4px;overflow:hidden}
.prob-bar{height:100%;border-radius:4px;width:0;transition:width 1s cubic-bezier(.4,0,.2,1)}
.pb-h{background:linear-gradient(90deg,#7f1d1d,var(--h))}
.pb-s{background:linear-gradient(90deg,#064e3b,var(--g))}
.prob-val{font-family:var(--fm);font-size:11px;min-width:58px;text-align:right}
.thr-row{display:flex;flex-direction:column;gap:5px;margin-top:8px}
.thr-labels{display:flex;justify-content:space-between;font-family:var(--fm);font-size:8px;color:var(--t4)}
.thr-track{position:relative;height:5px;background:linear-gradient(90deg,rgba(16,185,129,.3),rgba(244,63,94,.3));border-radius:3px}
.thr-dot{position:absolute;top:50%;width:11px;height:11px;border-radius:50%;background:var(--a);border:2px solid var(--bg);transform:translate(-50%,-50%);box-shadow:0 0 8px rgba(245,158,11,.8);transition:left .5s}
.thr-lbl{font-family:var(--fm);font-size:9px;color:var(--a);text-align:center;margin-top:3px}

/* Spans */
.span-head{font-family:var(--fm);font-size:9px;letter-spacing:2.5px;text-transform:uppercase;color:var(--t4);margin-bottom:8px}
.spans-content{display:flex;flex-wrap:wrap;gap:6px}
.span-chip{font-family:var(--fm);font-size:11px;font-weight:500;padding:4px 14px;border-radius:6px;background:rgba(244,63,94,.1);border:1px solid rgba(244,63,94,.25);color:var(--h)}
.no-spans{font-family:var(--fm);font-size:10px;color:var(--t4)}

/* Highlight & BIO */
.hl-box{font-family:var(--fb);font-size:14px;line-height:2.1;background:rgba(10,16,40,.5);border:1px solid var(--border);border-radius:var(--r);padding:16px 18px}
.tok-b{background:rgba(244,63,94,.18);border:1px solid rgba(244,63,94,.3);color:var(--h);border-radius:4px;padding:0 4px;margin:0 1px}
.tok-i{background:rgba(244,63,94,.1);border:1px solid rgba(244,63,94,.18);color:#fca5a5;border-radius:4px;padding:0 4px;margin:0 1px}
.tok-o{color:var(--t2)}
.bio-tbl{width:100%;border-collapse:collapse;font-size:12px}
.bio-tbl th{font-family:var(--fm);font-size:8.5px;letter-spacing:2px;text-transform:uppercase;color:var(--t4);padding:7px 12px;border-bottom:1px solid var(--border);text-align:left}
.bio-tbl td{padding:6px 12px;border-bottom:1px solid rgba(255,255,255,.03);color:var(--t2);font-family:var(--fm)}
.bio-tbl tr:hover td{background:rgba(124,58,237,.04)}
.tag-b{background:rgba(244,63,94,.15);color:var(--h);border:1px solid rgba(244,63,94,.3);padding:2px 9px;border-radius:5px;font-size:10px}
.tag-i{background:rgba(251,113,133,.1);color:#fca5a5;border:1px solid rgba(251,113,133,.2);padding:2px 9px;border-radius:5px;font-size:10px}
.tag-o{background:rgba(255,255,255,.05);color:var(--t4);border:1px solid var(--border);padding:2px 9px;border-radius:5px;font-size:10px}

/* LIME vis */
.lime-vis{min-height:200px;background:rgba(5,10,20,.7);border:1px solid var(--border);border-radius:var(--r2);overflow:hidden;display:flex;align-items:center;justify-content:center}
.lime-vis img{width:100%;height:auto;display:block}
.loading-inner{display:flex;flex-direction:column;align-items:center;gap:14px;color:var(--t3)}
.spinner{width:34px;height:34px;border:2px solid rgba(124,58,237,.2);border-top-color:var(--v2);border-radius:50%;animation:spin .8s linear infinite}
.loading-inner span{font-family:var(--fm);font-size:10px;letter-spacing:2px}
.lime-sum{font-family:var(--fm);font-size:10px;line-height:1.9;color:var(--t3);background:rgba(10,16,40,.7);border:1px solid var(--border);border-radius:var(--r);padding:14px 16px;margin-top:10px;white-space:pre-line;display:none}
.lime-sum.show{display:block;animation:paneIn .3s ease}
@keyframes paneIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}

/* Results enter animation */
.rp-inner>*{animation:paneIn .6s cubic-bezier(.16,1,.3,1) both}
.rp-inner>*:nth-child(1){animation-delay:.06s}
.rp-inner>*:nth-child(2){animation-delay:.13s}
.rp-inner>*:nth-child(3){animation-delay:.20s}
.rp-inner>*:nth-child(4){animation-delay:.27s}
.rp-inner>*:nth-child(5){animation-delay:.34s}
.rp-inner>*:nth-child(6){animation-delay:.41s}
.rp-inner>*:nth-child(7){animation-delay:.48s}
.rp-inner>*:nth-child(8){animation-delay:.55s}

/* ─── SKELETON LOADING ─── */
@keyframes shimmer{
  0%{background-position:200% center}
  100%{background-position:-200% center}
}
.sk-line{
  border-radius:6px;
  background:linear-gradient(90deg,rgba(255,255,255,.04) 25%,rgba(167,139,250,.12) 50%,rgba(255,255,255,.04) 75%);
  background-size:400% 100%;
  animation:shimmer 1.8s ease-in-out infinite;
}
.sk-h-lg{height:40px;margin-bottom:10px}
.sk-h-md{height:22px;margin-bottom:8px}
.sk-h-sm{height:14px;margin-bottom:6px}
.sk-w-full{width:100%}
.sk-w-3q{width:75%}
.sk-w-half{width:50%}
.sk-w-1q{width:30%}
.sk-circle{border-radius:50%;width:72px;height:72px;flex-shrink:0}
.sk-card{background:var(--glass);border:1px solid var(--border);border-radius:var(--r2);padding:20px 22px;margin-bottom:14px;display:flex;align-items:center;gap:16px}
.sk-block{background:var(--glass);border:1px solid var(--border);border-radius:var(--r2);padding:16px 18px;margin-bottom:14px}
.sk-rows{display:flex;flex-direction:column;flex:1}

/* Textarea pulse on analyze */
@keyframes taPulse{
  0%{box-shadow:0 0 0 0 rgba(124,58,237,.0),0 0 0 3px rgba(124,58,237,.08)}
  40%{box-shadow:0 0 0 6px rgba(124,58,237,.18),0 0 0 3px rgba(124,58,237,.08)}
  100%{box-shadow:0 0 0 0 rgba(124,58,237,.0),0 0 0 3px rgba(124,58,237,.08)}
}
.ta-wrap.ta-analyzing{animation:taPulse .55s ease-out forwards}

/* Status bar — smoother state transitions */
.sbar{display:flex;align-items:center;gap:9px;font-family:var(--fm);font-size:11px;padding:10px 14px;background:rgba(6,10,24,.5);border-radius:10px;border:1px solid transparent;color:var(--t4);transition:color .35s ease,border-color .35s ease,background .35s ease}
.sb-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0;transition:background .35s ease,box-shadow .35s ease}

/* Result content fade-in */
@keyframes resultFadeIn{
  from{opacity:0;transform:translateY(8px)}
  to{opacity:1;transform:none}
}
.result-reveal{animation:resultFadeIn .45s cubic-bezier(.16,1,.3,1) both}
</style>
</head>
<body>

<!-- SPLASH -->
<div id="splash">
  <canvas id="pCanvas"></canvas>
  <div class="sp-orb sp-orb1"></div>
  <div class="sp-orb sp-orb2"></div>
  <div class="sp-orb sp-orb3"></div>
  <div class="rings">
    <div class="ring ring1"></div>
    <div class="ring ring2"></div>
    <div class="ring ring3"></div>
    <div class="ring ring4"></div>
  </div>
  <div class="sp-inner">
    <div class="sp-eyebrow">BLOOM-560M MTL &nbsp;·&nbsp; THESIS LIVE DEMO</div>
    <h1 class="sp-title">HATE SPEECH<br>DETECTOR</h1>
    <p class="sp-sub">Sistem Deteksi Ujaran Kebencian Berbasis<br>Large Language Model dengan Token Labeling CRF &amp; LIME</p>
    <p class="sp-thesis">SKRIPSI / THESIS DEMO &nbsp;·&nbsp; {{ device_label }} &nbsp;·&nbsp; θ = {{ threshold }}</p>
    <div class="sp-press" id="spBtn">
      <div class="press-line"></div>
      <span>PRESS ANY KEY TO ENTER</span>
      <div class="press-line"></div>
    </div>
    <div class="sp-bracket tl"></div><div class="sp-bracket tr"></div>
    <div class="sp-bracket bl"></div><div class="sp-bracket br"></div>
  </div>
</div>

<!-- TRANSITION -->
<div id="transSeq" aria-hidden="true">
  <div class="scan-h" id="scanLine"></div>
  <div class="init-txt" id="initTxt">INITIALIZING</div>
</div>

<!-- APP -->
<div id="app">
  <div class="bg-wrap">
    <div class="orb o1"></div><div class="orb o2"></div>
    <div class="orb o3"></div><div class="orb o4"></div>
    <div class="grid-overlay"></div>
  </div>

  <div class="ws">
    <!-- LEFT / CENTER PANEL -->
    <div class="lp">

      <!-- Split-mode brand (visible only after split) -->
      <div class="sp-brand">
        <div class="sp-brand-dot"></div>
        Hate Speech <em>Detector</em>
        <span class="sp-brand-sub">BLOOM-560M MTL</span>
      </div>

      <!-- Center-mode brand (visible in default state) -->
      <div class="cs-brand">
        <div class="cs-eyebrow">BLOOM-560M MTL</div>
        <div class="cs-title">Hate Speech <em>Detector</em></div>
        <div class="cs-sub">Deteksi Ujaran Kebencian Berbasis Large Language Model</div>
      </div>

      <!-- Input -->
      <div class="ta-wrap">
        <div class="ta-scanline"></div>
        <textarea id="inputText" placeholder="Ketik teks untuk dianalisis — Bahasa Indonesia, English, atau campuran keduanya...&#10;&#10;Enter untuk analisis  ·  Shift+Enter untuk baris baru."></textarea>
      </div>

      <!-- Buttons -->
      <div class="btn-row">
        <button class="btn-p" id="btnAnalyze" onclick="runAnalyze()">
          <svg width="13" height="13" fill="none" viewBox="0 0 12 12" style="position:relative;z-index:1"><circle cx="6" cy="6" r="4.5" stroke="currentColor" stroke-width="1.2"/><path d="M6 3.5v2.5l1.5 1.5" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/></svg>
          ANALYZE
        </button>
        <button class="btn-s" onclick="clearAll()">CLEAR</button>
        <span class="char-ct" id="charCt">0 / 512</span>
      </div>

      <!-- LIME (hidden in center, revealed after split) -->
      <div class="lime-sect">
        <div class="panel-lbl"><div class="panel-lbl-dot" style="background:var(--c);box-shadow:0 0 6px var(--c)"></div>LIME INTERPRETABILITY</div>
        <div class="lime-ctrl">
          <span class="s-lbl">SAMPLES</span>
          <input type="range" id="limeSlider" min="50" max="500" value="200" step="50"
            oninput="this.style.setProperty('--p',((this.value-50)/450*100)+'%');document.getElementById('limeVal').textContent=this.value">
          <span class="s-val" id="limeVal">200</span>
          <button class="btn-s" id="btnLime" onclick="runLime()">RUN LIME</button>
        </div>
        <div class="lime-cav">⚠ LIME beroperasi di level kata — BPE tokenizer BLOOM menyebabkan mismatch subword. Gunakan sebagai indikasi kualitatif.</div>
      </div>

      <!-- Status bar -->
      <div class="sbar s-idle" id="statusBar">
        <div class="sb-dot"></div>
        <span id="statusTxt">STANDBY — masukkan teks lalu tekan Enter</span>
      </div>

    </div><!-- .lp -->

    <!-- RIGHT PANEL (results) -->
    <div class="rp" id="rp">
      <div class="rp-inner" id="rpInner">

        <!-- Verdict -->
        <div class="gc verdict is-neutral" id="verdictCard">
          <div class="vring r1 rn" id="vr1"></div>
          <div class="vring r2 rn" id="vr2"></div>
          <div class="vring r3 rn" id="vr3"></div>
          <div class="v-layout">
            <div class="v-icon" id="vIcon">
              <svg viewBox="0 0 52 52" fill="none"><circle cx="26" cy="26" r="22" stroke="rgba(160,185,255,.18)" stroke-width="1.2" opacity=".4"/><circle cx="26" cy="26" r="6" stroke="rgba(160,185,255,.22)" stroke-width="1.5"/></svg>
            </div>
            <div class="v-text">
              <div class="v-label">VERDICT</div>
              <div class="v-big neutral" id="vBig">STANDBY</div>
              <div class="v-sub" id="vSub" style="color:var(--t4)">Masukkan teks dan tekan Enter untuk menganalisis</div>
            </div>
            <div class="conf-wrap">
              <svg class="conf-svg" viewBox="0 0 76 76">
                <circle class="conf-bg" cx="38" cy="38" r="33"/>
                <circle class="conf-fg" id="confRing" cx="38" cy="38" r="33" style="stroke:rgba(160,185,255,.18);stroke-dashoffset:207"/>
              </svg>
              <div class="conf-center">
                <div class="conf-num" id="confPct" style="color:rgba(160,185,255,.28)">—</div>
                <div class="conf-lbl-sm">CONF</div>
              </div>
            </div>
          </div>
        </div>

        <!-- Prob bars -->
        <div class="gc" style="padding:16px 18px">
          <div class="prob-row"><div class="prob-lbl" style="color:var(--h)">HATE</div><div class="prob-bar-bg"><div class="prob-bar pb-h" id="phBar" style="width:0%"></div></div><div class="prob-val" id="phV" style="color:var(--h)">—</div></div>
          <div class="prob-row" style="margin-bottom:0"><div class="prob-lbl" style="color:var(--g)">SAFE</div><div class="prob-bar-bg"><div class="prob-bar pb-s" id="psBar" style="width:0%"></div></div><div class="prob-val" id="psV" style="color:var(--g)">—</div></div>
          <div class="thr-row">
            <div class="thr-labels"><span>0</span><span>THRESHOLD</span><span>1</span></div>
            <div class="thr-track"><div class="thr-dot" id="thrDot" style="left:44.29%"></div></div>
            <div class="thr-lbl" id="thrLbl">θ = {{ threshold }}</div>
          </div>
        </div>

        <!-- Toxic Spans -->
        <div class="sec-lbl-row">
          <div class="panel-lbl-dot" style="background:var(--h);box-shadow:0 0 6px var(--h)"></div>TOXIC SPANS
        </div>
        <div class="gc" style="padding:14px 18px;margin-bottom:14px">
          <div class="span-head">Kata / frasa terdeteksi toxic:</div>
          <div class="spans-content">
            <span class="no-spans">—</span>
          </div>
        </div>

        <!-- Highlighted text -->
        <div class="sec-lbl-row">
          <div class="panel-lbl-dot"></div>HIGHLIGHTED TEXT
        </div>
        <!-- hlSection: BUKAN .gc — build_highlighted_html() sudah menyediakan
             border/padding/border-radius sendiri. Menambahkan .gc di sini akan
             menyebabkan double border + double padding (BUG DIPERBAIKI). -->
        <div id="hlSection" style="margin-bottom:14px">
          <div style="font-family:var(--fm);font-size:10px;color:var(--t4);
            padding:14px 16px;background:rgba(8,14,36,0.4);
            border:1px solid rgba(255,255,255,0.07);border-radius:12px;">—</div>
        </div>

        <!-- BIO Tags -->
        <div class="sec-lbl-row">
          <div class="panel-lbl-dot" style="background:var(--c);box-shadow:0 0 6px var(--c)"></div>BIO TAGS
        </div>
        <div class="gc" style="overflow-x:auto;margin-bottom:14px" id="bioSection">
          <div style="font-family:var(--fm);font-size:10px;color:var(--t4);padding:16px 18px;">—</div>
        </div>

        <!-- LIME -->
        <div class="sec-lbl-row">
          <div class="panel-lbl-dot" style="background:var(--a);box-shadow:0 0 6px var(--a)"></div>LIME INTERPRETABILITY
        </div>
        <div id="limeSection">
          <div class="lime-vis">
            <div style="font-family:var(--fm);font-size:10px;color:var(--t4);text-align:center;padding:24px">
              Jalankan LIME dari panel kiri<br>untuk melihat kontribusi kata.
            </div>
          </div>
        </div>

      </div><!-- .rp-inner -->
    </div><!-- .rp -->
  </div><!-- .ws -->
</div><!-- #app -->

<script>
/* CANVAS PARTICLES */
(function(){
  const cv=document.getElementById('pCanvas'),ctx=cv.getContext('2d');
  let pts=[],W,H,raf,alive=true;
  const C=['rgba(167,139,250,','rgba(103,232,249,','rgba(196,120,255,'];
  function resize(){W=cv.width=window.innerWidth;H=cv.height=window.innerHeight}
  function init(){
    resize();pts=[];
    const N=Math.min(Math.floor(W*H/13000),110);
    for(let i=0;i<N;i++) pts.push({x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-.5)*.42,vy:(Math.random()-.5)*.42,r:Math.random()*1.4+.4,op:Math.random()*.5+.1,c:C[Math.floor(Math.random()*C.length)]});
  }
  function draw(){
    if(!alive)return;
    ctx.clearRect(0,0,W,H);
    for(let i=0;i<pts.length;i++){
      for(let j=i+1;j<pts.length;j++){
        const dx=pts[i].x-pts[j].x,dy=pts[i].y-pts[j].y,d=Math.hypot(dx,dy);
        if(d<130){ctx.beginPath();ctx.moveTo(pts[i].x,pts[i].y);ctx.lineTo(pts[j].x,pts[j].y);ctx.strokeStyle=`rgba(139,124,248,${(1-d/130)*.11})`;ctx.lineWidth=.5;ctx.stroke()}
      }
    }
    for(let p of pts){
      ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fillStyle=p.c+p.op+')';ctx.fill();
      p.x+=p.vx;p.y+=p.vy;
      if(p.x<0)p.x=W;if(p.x>W)p.x=0;if(p.y<0)p.y=H;if(p.y>H)p.y=0;
    }
    raf=requestAnimationFrame(draw);
  }
  window.addEventListener('resize',()=>resize());
  init();draw();
  window._ptsBurst=function(){
    pts.forEach(p=>{
      const cx=W/2,cy=H/2,dx=p.x-cx,dy=p.y-cy,d=Math.hypot(dx,dy)||1;
      p.vx+=(dx/d)*4*(Math.random()+.4);p.vy+=(dy/d)*4*(Math.random()+.4);
      setTimeout(()=>{p.vx=(Math.random()-.5)*.42;p.vy=(Math.random()-.5)*.42},900);
    });
  };
  window._stopParticles=function(){alive=false;cancelAnimationFrame(raf)};
})();

/* TRANSITION */
let _entered=false;
function triggerEntry(){
  if(_entered)return;_entered=true;
  if(window._ptsBurst)window._ptsBurst();

  /* 1 — blur-out & scale the entire splash inner content */
  const inner=document.querySelector('.sp-inner');
  inner.style.transition='filter .3s ease, transform .3s ease, opacity .28s ease';
  inner.style.filter='blur(12px)';
  inner.style.transform='scale(1.04) translateY(-14px)';
  inner.style.opacity='0';

  /* 2 — begin splash collapse (slightly delayed so blur starts first) */
  setTimeout(()=>{document.getElementById('splash').classList.add('vanish');},220);

  /* 3 — scan line fires while splash is mid-collapse */
  setTimeout(()=>{
    const ts=document.getElementById('transSeq');
    ts.classList.add('active');
    const sh=document.getElementById('scanLine');
    sh.style.animation='none';void sh.offsetWidth;
    sh.style.animation='scanDn .8s cubic-bezier(.4,0,.6,1) forwards';
  },360);

  /* 4 — reveal initTxt via opacity */
  setTimeout(()=>{
    const it=document.getElementById('initTxt');
    if(it) requestAnimationFrame(()=>{ it.classList.add('visible'); });
  },660);

  /* 5 — fade out transition overlay + reveal app with upward slide */
  setTimeout(()=>{
    document.getElementById('transSeq').classList.add('done');
    document.getElementById('app').classList.add('show');
    setTimeout(()=>{
      const s=document.getElementById('splash');if(s)s.remove();
      if(window._stopParticles)window._stopParticles();
    },1600);
  },1180);
}
document.addEventListener('keydown',()=>triggerEntry(),{once:true});
document.getElementById('splash').addEventListener('click',()=>triggerEntry(),{once:true});
document.getElementById('spBtn').addEventListener('click',()=>triggerEntry(),{once:true});

/* APP LOGIC */
const $ta=document.getElementById('inputText');
$ta.addEventListener('input',()=>{
  document.getElementById('charCt').textContent=$ta.value.length+' / 512';
});

function setStatus(msg,cls){
  document.getElementById('statusBar').className='sbar '+cls;
  document.getElementById('statusTxt').textContent=msg;
}

function clearAll(){
  $ta.value='';
  document.getElementById('charCt').textContent='0 / 512';
  setStatus('STANDBY — masukkan teks lalu tekan Enter','s-idle');
  /* Collapse back to center if desired */
  /* Uncomment next line to reset to center on clear: */
  /* document.getElementById('app').classList.remove('split'); */
}

/* Escape HTML untuk mencegah XSS pada token yang dirender di BIO table */
function escHtml(s){
  return String(s)
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}

function renderResult(d){
  const isHate = d.pred === 'HATE SPEECH';

  /* Hide loading overlay first */
  hideRpOverlay();

  /* Ensure split layout */
  document.getElementById('app').classList.add('split');

  /* Re-trigger stagger animations on right panel children */
  const inner = document.getElementById('rpInner');
  Array.from(inner.children).forEach(el=>{
    el.style.animation='none';
    void el.offsetWidth;
    el.style.animation='';
  });

  /* Verdict card */
  const card=document.getElementById('verdictCard');
  card.classList.remove('is-neutral');
  card.classList.toggle('is-hate',isHate);
  card.classList.toggle('is-safe',!isHate);

  /* Rings */
  ['vr1','vr2','vr3'].forEach(id=>{
    const el=document.getElementById(id);
    el.className='vring '+{vr1:'r1',vr2:'r2',vr3:'r3'}[id]+' '+(isHate?'rh':'rs');
  });

  /* Icon */
  const vIcon=document.getElementById('vIcon');
  if(isHate){
    vIcon.innerHTML=`<svg viewBox="0 0 52 52" fill="none"><circle cx="26" cy="26" r="22" stroke="#f43f5e" stroke-width="1.2" opacity=".4"/><path d="M18 18l16 16M34 18L18 34" stroke="#f43f5e" stroke-width="2.5" stroke-linecap="round"/></svg>`;
  } else {
    vIcon.innerHTML=`<svg viewBox="0 0 52 52" fill="none"><circle cx="26" cy="26" r="22" stroke="#10b981" stroke-width="1.2" opacity=".4"/><path d="M16 26l8 8 13-13" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  }

  /* Verdict text */
  const vBig=document.getElementById('vBig');
  vBig.textContent=isHate?'HATE SPEECH':'CLEAN';
  vBig.className='v-big '+(isHate?'hate':'safe');

  /* Sub */
  const vSub=document.getElementById('vSub');
  vSub.style.color='';
  vSub.textContent=isHate
    ?'HATE SPEECH — Teks mengandung ujaran kebencian'
    :'NON-HATE SPEECH — Teks tidak mengandung ujaran kebencian';

  /* Confidence ring — reset then animate to final value */
  const ringColor=isHate?'#f43f5e':'#10b981';
  const confRing=document.getElementById('confRing');
  confRing.style.transition='none';
  confRing.style.strokeDashoffset='207';
  confRing.style.stroke=ringColor;
  document.getElementById('confPct').textContent='';
  document.getElementById('confPct').style.color=ringColor;
  void confRing.offsetWidth;
  confRing.style.transition='stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1),stroke .5s';
  confRing.style.strokeDashoffset=207-(207*d.conf);
  setTimeout(()=>{
    document.getElementById('confPct').textContent=Math.round(d.conf*100)+'%';
    document.getElementById('confPct').classList.add('result-reveal');
    setTimeout(()=>document.getElementById('confPct').classList.remove('result-reveal'),500);
  },400);

  /* Prob bars — animate from 0 */
  const phBar=document.getElementById('phBar');
  const psBar=document.getElementById('psBar');
  phBar.style.transition='none';psBar.style.transition='none';
  phBar.style.width='0%';psBar.style.width='0%';
  void phBar.offsetWidth;
  phBar.style.transition='width 1.1s cubic-bezier(.4,0,.2,1)';
  psBar.style.transition='width 1.1s cubic-bezier(.4,0,.2,1) .08s';
  phBar.style.width=Math.max(4,Math.round(d.prob_hate*100))+'%';
  document.getElementById('phV').textContent=d.prob_hate.toFixed(4);
  psBar.style.width=Math.max(4,Math.round(d.prob_nonhate*100))+'%';
  document.getElementById('psV').textContent=d.prob_nonhate.toFixed(4);

  /* Threshold */
  document.getElementById('thrDot').style.left=(d.threshold*100).toFixed(2)+'%';
  document.getElementById('thrLbl').textContent='θ = '+d.threshold.toFixed(4);

  /* Spans */
  const spansEl=document.querySelector('.spans-content');
  if(spansEl){
    spansEl.innerHTML=d.spans&&d.spans.length
      ?d.spans.map(s=>`<span class="span-chip result-reveal">${s}</span>`).join('')
      :'<span class="no-spans result-reveal">Tidak ada toxic span terdeteksi.</span>';
  }

  /* Highlight */
  /* build_highlighted_html() sudah mengembalikan outer <div> dengan styling lengkap.
     Tidak perlu .hl-box wrapper tambahan — akan menyebabkan double border/padding. */
  document.getElementById('hlSection').innerHTML=`<div class="result-reveal">${d.highlight_html}</div>`;

  /* BIO table — tag matching is flexible (B-TOX/I-TOX, B/I, B-HATE/I-HATE, etc.) */
  const tbody=d.bio_rows.map(r=>{
    const tag=String(r[2]).toUpperCase();
    const cls=tag.startsWith('B')?'tag-b':tag.startsWith('I')?'tag-i':'tag-o';
    return `<tr><td>${escHtml(r[0])}</td><td>${escHtml(r[1])}</td><td><span class="${cls}">${escHtml(r[2])}</span></td><td>${escHtml(r[3])}</td></tr>`;
  }).join('');
  document.getElementById('bioSection').innerHTML=
    `<table class="bio-tbl result-reveal"><thead><tr><th>#</th><th>Token</th><th>BIO</th><th>Keterangan</th></tr></thead><tbody>${tbody}</tbody></table>`;

  /* Debug — buka DevTools Console untuk lihat format aktual */
  if(d.bio_rows&&d.bio_rows.length) console.debug('[BIO] sample tag format:', d.bio_rows[0]);
}

function showRpOverlay(){
  let ov=document.getElementById('rpOverlay');
  if(!ov){
    ov=document.createElement('div');
    ov.id='rpOverlay';
    ov.className='rp-overlay';
    ov.innerHTML='<div class="rp-overlay-inner"><div class="spinner"></div><span>MENGANALISIS...</span></div>';
    document.getElementById('rp').appendChild(ov);
  }
  ov.style.display='flex';
  void ov.offsetWidth;
  ov.classList.add('visible');
}

function hideRpOverlay(){
  const ov=document.getElementById('rpOverlay');
  if(!ov)return;
  ov.classList.remove('visible');
  setTimeout(()=>{ov.style.display='none';},280);
}

async function runAnalyze(){
  const text=$ta.value.trim();
  if(!text){setStatus('Teks kosong.','s-err');return;}

  /* Textarea pulse feedback */
  const taWrap=document.querySelector('.ta-wrap');
  taWrap.classList.remove('ta-analyzing');
  void taWrap.offsetWidth;
  taWrap.classList.add('ta-analyzing');
  setTimeout(()=>taWrap.classList.remove('ta-analyzing'),600);

  setStatus('Menganalisis teks...','s-load');
  document.getElementById('btnAnalyze').disabled=true;

  /* Open split panel immediately, then show overlay on top (DOM stays intact) */
  const appEl=document.getElementById('app');
  if(!appEl.classList.contains('split')){
    appEl.classList.add('split');
    /* Wait for panel to start opening before showing overlay */
    setTimeout(()=>showRpOverlay(), 300);
  } else {
    showRpOverlay();
  }

  try{
    const res=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
    const d=await res.json();
    if(!res.ok){hideRpOverlay();setStatus(d.error||'Error','s-err');return;}
    renderResult(d);
    setStatus(d.pred+' · P(hate)='+d.prob_hate.toFixed(4),'s-ok');
  }catch(e){
    hideRpOverlay();
    setStatus('Error: '+e.message,'s-err');
  }
  finally{document.getElementById('btnAnalyze').disabled=false;}
}

async function runLime(){
  const text=$ta.value.trim();
  if(!text){setStatus('Teks kosong.','s-err');return;}
  const nSamples=parseInt(document.getElementById('limeSlider').value)||200;
  setStatus('LIME sedang berjalan — harap tunggu...','s-load');
  document.getElementById('btnLime').disabled=true;
  document.getElementById('limeSection').innerHTML=
    `<div class="lime-vis"><div class="loading-inner"><div class="spinner"></div><span>COMPUTING ${nSamples} PERTURBATIONS...</span></div></div>`;
  document.getElementById('limeSection').scrollIntoView({behavior:'smooth',block:'start'});
  try{
    const res=await fetch('/api/lime',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,n_samples:nSamples})});
    const d=await res.json();
    if(!res.ok){
      setStatus(d.error||'LIME error','s-err');
      document.getElementById('limeSection').innerHTML=`<div class="lime-vis" style="padding:20px;color:var(--h);font-family:var(--fm);font-size:11px;">${d.error||'LIME gagal'}</div>`;
      return;
    }
    const top3txt=d.top3.map(([w,v])=>`  ${w.padEnd(18)} ${v>0?'+':''}${v.toFixed(4)}`).join('\n');
    document.getElementById('limeSection').innerHTML=
      `<div class="lime-vis"><img src="data:image/png;base64,${d.image_b64}" alt="LIME chart"/></div>
       <div class="lime-sum show">MODEL: ${d.pred_label}  |  CONF: ${(d.confidence*100).toFixed(1)}%  |  SAMPLES: ${d.n_samples}\n\nTop kontribusi kata:\n${top3txt}</div>`;
    setStatus('LIME selesai · '+d.pred_label,'s-ok');
  }catch(e){setStatus('LIME error: '+e.message,'s-err');}
  finally{document.getElementById('btnLime').disabled=false;}
}

$ta.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();runAnalyze();}
});
</script>
</body>
</html>
"""



# ══


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Hate Speech Demo — Flask")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--bloom",      default=DEFAULT_BLOOM_PATH)
    parser.add_argument("--threshold",  default=DEFAULT_THRESHOLD_JSON)
    parser.add_argument("--lime-samples", type=int, default=DEFAULT_LIME_SAMPLES)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
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
        for m in missing:
            print(m)
        print("\nIkuti langkah di DEMO_GUIDE.md untuk mengunduh model.")
        print("="*60)
        sys.exit(1)

    print("\n" + "="*60)
    print(" Hate Speech Demo — BLOOM-560m MTL")
    print("="*60)
    _initialize(args.checkpoint, args.bloom, args.threshold)
    print(f"[App] Buka browser di http://localhost:{args.port}")
    print("="*60)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()