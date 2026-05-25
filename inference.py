"""
inference.py — Prediksi teks + token-level BIO tagging
=======================================================
Fungsi predict_text() mereplikasi predict_single_v2() dari NB05v2 secara
identik: offset_mapping, char-to-word alignment, CRF Viterbi, span extraction.

Fungsi build_*() menghasilkan output Gradio (HTML highlight + tabel BIO)
dalam dark/glassmorphism theme yang cocok dengan UI futuristik.

BUGFIX yang diterapkan:
  - I-TOX word_bio assignment DIPERBAIKI: sebelumnya salah mempromosi
    I-TOX → B-TOX (merusak span multi-kata). Kini identik NB05v2.
  - char_to_word space mapping documented: maps space position to the
    immediately following word index (intentional, matches NB05v2).
"""

import re
import torch
import numpy as np
from model_utils import ID2LABEL


# ══════════════════════════════════════════════════════════════════════════
#  Fungsi prediksi utama
# ══════════════════════════════════════════════════════════════════════════
def predict_text(
    text:       str,
    tokenizer,
    model,
    device:     torch.device,
    threshold:  float = 0.4429,
    max_length: int   = 512,
) -> dict:
    """
    Menerima teks bebas dan mengembalikan:
      - pred        : "HATE SPEECH" atau "NON-HATE"
      - prob_hate   : float 0–1
      - prob_nonhate: float 0–1
      - conf        : confidence prediksi
      - words       : list[str]
      - word_bio    : list["O"|"B-TOX"|"I-TOX"]
      - spans       : list[str]  token-span toxic

    Alur identik dengan predict_single_v2() di NB05v2:
      1. Normalisasi whitespace
      2. Tokenisasi dengan return_offsets_mapping=True
      3. Forward pass → sentence_logits + token_preds_crf
      4. Temperature scaling pada sentence logits
      5. Char→word alignment via offset_map
      6. Rekonstruksi word_bio & span extraction
    """
    if not text or not text.strip():
        return _empty_result()

    model.eval()

    # ── Step 1: Normalisasi ─────────────────────────────────────────────
    text_norm = re.sub(r"\s+", " ", str(text)).strip()
    if not text_norm:
        return _empty_result()
    words = text_norm.split()

    # ── Step 2: Tokenisasi ──────────────────────────────────────────────
    enc = tokenizer(
        text_norm,
        padding                = "max_length",
        truncation             = True,
        max_length             = max_length,
        return_tensors         = "pt",
        return_offsets_mapping = True,
    )
    # offset_mapping HARUS di-pop sebelum model.forward()
    # karena BLOOM tidak menerima argument ini
    offset_map    = enc.pop("offset_mapping")[0].tolist()
    attn_mask_cpu = enc["attention_mask"].clone()

    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # ── Step 3: Forward pass ────────────────────────────────────────────
    with torch.no_grad():
        out = model(input_ids, attention_mask)

    # ── Step 4: Temperature scaling (identik collect_predictions NB05) ─
    # CATATAN AKADEMIS: predict_single_v2 di NB05v2 (page 308) TIDAK menerapkan
    # temperature — kemungkinan oversight. Demo ini LEBIH KONSISTEN karena
    # BEST_THRESHOLD_V2 = 0.4429 dikalibrasi via collect_predictions yang
    # MENGGUNAKAN temperature (NB05v2 page 242). Dengan T=0.9710 (main model),
    # probabilitas sedikit lebih tajam dan presisi threshold terjaga.
    temperature  = float(getattr(model, "temperature", 1.0))
    proba        = torch.softmax(
        out["sentence_logits"].float() / temperature, dim=-1
    )[0]
    prob_hate    = proba[1].item()
    prob_nonhate = proba[0].item()

    pred = "HATE SPEECH" if prob_hate >= threshold else "NON-HATE"
    conf = prob_hate if prob_hate >= threshold else prob_nonhate

    # ── Step 5: Char→word alignment ─────────────────────────────────────
    # token_preds_crf is a list[list[int]] from CRF Viterbi; take batch[0]
    crf_preds         = out["token_preds_crf"][0]
    attn_list         = attn_mask_cpu[0].tolist()
    non_pad_positions = [i for i, m in enumerate(attn_list) if m == 1]

    # Build char→word map.
    # Space at (pos-1) is mapped to the NEXT word (w_idx) — intentional,
    # matches NB05v2 behaviour when a BPE token spans across a word boundary.
    char_to_word: dict = {}
    pos = 0
    for w_idx, w in enumerate(words):
        if w_idx > 0:
            char_to_word[pos - 1] = w_idx   # space before this word → next word
        for c in range(len(w)):
            char_to_word[pos + c] = w_idx
        pos += len(w) + 1

    word_bio = ["O"] * len(words)
    for crf_label, tok_idx in zip(crf_preds, non_pad_positions):
        start, end = offset_map[tok_idx]
        if start == end:     # padding / special token (zero-length span)
            continue
        widx = char_to_word.get(start)
        if widx is None and start > 0:
            widx = char_to_word.get(start - 1)
        if widx is None or widx >= len(words):
            continue
        tag = ID2LABEL[crf_label]
        if tag == "B-TOX":
            word_bio[widx] = "B-TOX"
        elif tag == "I-TOX" and word_bio[widx] == "O":
            # Identik dengan predict_single_v2 NB05v2:
            #   elif tag == 'I-TOX' and word_bio[widx] == 'O':
            #       word_bio[widx] = 'I-TOX'
            # PENTING: jangan promosi ke B-TOX — ini merusak span extraction
            # multi-kata. Contoh: "anjing tua" → B-TOX I-TOX harus tetap
            # satu span ['anjing tua'], bukan dua span ['anjing', 'tua'].
            # CRF Viterbi dengan transisi valid (torchcrf) tidak akan
            # menghasilkan I-TOX tanpa B-TOX sebelumnya dalam satu kalimat,
            # tetapi pada level kata (word-level alignment), subword pertama
            # sebuah kata yang melanjutkan span dapat ber-tag I-TOX — dan
            # harus tetap I-TOX agar span extraction benar.
            word_bio[widx] = "I-TOX"
            # else: word already tagged B-TOX or I-TOX from a prior subword;
            # no change needed — B-TOX takes priority over I-TOX.

    # ── Step 6: Span extraction — IDENTIK dengan predict_single_v2 NB05v2 ─
    # Struktur loop tepat sama dengan kode NB05v2 agar span extraction
    # konsisten secara akademis. I-TOX yang diikuti B-TOX sebelumnya
    # (cur non-empty) akan digabung ke span yang sama.
    spans: list = []
    cur:   list = []
    for w, bio in zip(words, word_bio):
        if bio == "B-TOX":
            if cur:
                spans.append(" ".join(cur))
            cur = [w]
        elif bio == "I-TOX" and cur:
            cur.append(w)
        else:
            if cur:
                spans.append(" ".join(cur))
            cur = []
    if cur:
        spans.append(" ".join(cur))

    return {
        "text_norm":    text_norm,
        "words":        words,
        "word_bio":     word_bio,
        "pred":         pred,
        "prob_hate":    round(prob_hate,    4),
        "prob_nonhate": round(prob_nonhate, 4),
        "conf":         round(conf,         4),
        "spans":        spans,
    }


def _empty_result() -> dict:
    return {
        "text_norm":    "",
        "words":        [],
        "word_bio":     [],
        "pred":         "—",
        "prob_hate":    0.0,
        "prob_nonhate": 0.0,
        "conf":         0.0,
        "spans":        [],
    }


# ══════════════════════════════════════════════════════════════════════════
#  Builder: HTML highlighted text  (dark theme)
# ══════════════════════════════════════════════════════════════════════════
def build_highlighted_html(words: list, word_bio: list) -> str:
    """
    Menghasilkan HTML teks dengan highlight dark-theme:
      • B-TOX → background merah neon + bold
      • I-TOX → background oranye neon
      • O      → teks normal
    """
    if not words:
        return (
            "<div style='padding:16px;font-family:JetBrains Mono,monospace;"
            "font-size:0.85em;color:rgba(0,245,255,0.3);'>[ — ]</div>"
        )

    parts = []
    for w, bio in zip(words, word_bio):
        safe_w = (w.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;"))
        if bio == "B-TOX":
            parts.append(
                f'<span style="'
                f'background:rgba(255,45,85,0.25);'
                f'color:#ff6680;'
                f'border:1px solid rgba(255,45,85,0.5);'
                f'font-weight:700;'
                f'padding:1px 6px;border-radius:4px;margin:0 2px;'
                f'font-family:JetBrains Mono,monospace;'
                f'box-shadow:0 0 8px rgba(255,45,85,0.3);'
                f'" title="B-TOX">{safe_w}</span>'
            )
        elif bio == "I-TOX":
            parts.append(
                f'<span style="'
                f'background:rgba(255,140,0,0.2);'
                f'color:#ffaa44;'
                f'border:1px solid rgba(255,140,0,0.4);'
                f'padding:1px 6px;border-radius:4px;margin:0 2px;'
                f'font-family:JetBrains Mono,monospace;'
                f'box-shadow:0 0 6px rgba(255,140,0,0.2);'
                f'" title="I-TOX">{safe_w}</span>'
            )
        else:
            parts.append(
                f'<span style="'
                f'color:rgba(220,238,255,0.75);'
                f'margin:0 2px;font-family:JetBrains Mono,monospace;'
                f'">{safe_w}</span>'
            )

    body   = " ".join(parts)
    legend = (
        '<div style="margin-top:10px;font-size:0.75em;'
        'color:rgba(200,220,255,0.4);font-family:JetBrains Mono,monospace;'
        'display:flex;gap:12px;align-items:center;">'
        '<span style="background:rgba(255,45,85,0.25);color:#ff6680;'
        'border:1px solid rgba(255,45,85,0.4);'
        'padding:1px 8px;border-radius:4px;font-weight:700;">B-TOX</span>'
        ' Awal span toxic &nbsp;&nbsp;'
        '<span style="background:rgba(255,140,0,0.2);color:#ffaa44;'
        'border:1px solid rgba(255,140,0,0.35);'
        'padding:1px 8px;border-radius:4px;">I-TOX</span>'
        ' Lanjutan span toxic'
        '</div>'
    )
    return (
        f'<div style="'
        f'font-size:1em;line-height:1.9;padding:14px 16px;'
        f'background:rgba(2,6,18,0.6);'
        f'border:1px solid rgba(0,245,255,0.1);'
        f'border-radius:12px;'
        f'backdrop-filter:blur(8px);'
        f'">{body}</div>{legend}'
    )


# ══════════════════════════════════════════════════════════════════════════
#  Builder: tabel BIO untuk Gradio Dataframe
# ══════════════════════════════════════════════════════════════════════════
def build_bio_table(words: list, word_bio: list) -> list:
    """
    Mengembalikan list of [No, Token, BIO, Keterangan] untuk gr.Dataframe.

    BUG FIX: semua kolom dikembalikan sebagai str agar cocok dengan
    datatype=["str","str","str","str"] di app.py (menghindari type mismatch
    antara str(i) dan datatype="number").
    """
    if not words:
        return [["—", "—", "—", "—"]]

    rows = []
    for i, (w, bio) in enumerate(zip(words, word_bio), start=1):
        if bio == "B-TOX":
            ket = "🔴 Awal span toxic"
        elif bio == "I-TOX":
            ket = "🟠 Lanjutan span toxic"
        else:
            ket = "✅ Aman"
        rows.append([str(i), w, bio, ket])
    return rows


# ══════════════════════════════════════════════════════════════════════════
#  Builder: verdict panel (dark/glass theme)
# ══════════════════════════════════════════════════════════════════════════
def build_verdict_html(
    pred:         str,
    prob_hate:    float,
    prob_nonhate: float,
    threshold:    float,
    spans:        list,
) -> str:
    """
    Menghasilkan HTML panel verdict dalam dark glassmorphism theme.
    Warna disesuaikan agar kontras baik di atas background gelap.
    """
    is_hate = pred == "HATE SPEECH"

    # ── Color scheme (dark-friendly) ────────────────────────────────────
    if is_hate:
        border_clr  = "rgba(255, 45, 85, 0.5)"
        bg_clr      = "rgba(255, 20, 60, 0.08)"
        label_clr   = "#ff4d6d"
        icon        = "⚠️"
        label_txt   = "HATE SPEECH"
        bar1_color  = "#ff2d55"
        bar2_color  = "rgba(0,255,136,0.4)"
        glow        = "0 0 30px rgba(255,45,85,0.25)"
    else:
        border_clr  = "rgba(0, 255, 136, 0.4)"
        bg_clr      = "rgba(0, 200, 100, 0.06)"
        label_clr   = "#00ff88"
        icon        = "✅"
        label_txt   = "NON-HATE"
        bar1_color  = "rgba(255,45,85,0.3)"
        bar2_color  = "#00cc6a"
        glow        = "0 0 30px rgba(0,255,136,0.15)"

    hate_pct    = max(4, int(prob_hate    * 100))
    nonhate_pct = max(4, int(prob_nonhate * 100))

    # ── Spans ────────────────────────────────────────────────────────────
    if spans:
        span_tags = "".join(
            f'<span style="'
            f'background:rgba(255,45,85,0.2);color:#ff6680;'
            f'border:1px solid rgba(255,45,85,0.45);'
            f'padding:3px 10px;border-radius:5px;'
            f'margin:3px;display:inline-block;'
            f'font-family:JetBrains Mono,monospace;font-size:0.88em;'
            f'box-shadow:0 0 8px rgba(255,45,85,0.2);'
            f'">{s}</span>'
            for s in spans
        )
        spans_html = (
            f'<div style="margin-top:14px;padding-top:12px;'
            f'border-top:1px solid rgba(255,255,255,0.07);">'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.72em;'
            f'color:rgba(255,100,120,0.7);letter-spacing:1.5px;'
            f'text-transform:uppercase;margin-bottom:8px;">◈ Toxic Span Terdeteksi</div>'
            f'{span_tags}</div>'
        )
    else:
        spans_html = (
            '<div style="margin-top:12px;padding-top:10px;'
            'border-top:1px solid rgba(255,255,255,0.06);'
            'font-family:JetBrains Mono,monospace;font-size:0.82em;'
            'color:rgba(200,230,255,0.35);font-style:italic;">'
            'Tidak ada toxic token terdeteksi.</div>'
        )

    return f"""
<div style="
    border:1px solid {border_clr};
    background:{bg_clr};
    backdrop-filter:blur(16px);
    -webkit-backdrop-filter:blur(16px);
    border-radius:14px;
    padding:20px 22px;
    font-family:'Syne',sans-serif;
    box-shadow:{glow},0 8px 32px rgba(0,0,0,0.4);
    position:relative;overflow:hidden;
">
  <!-- Decorative corner -->
  <div style="position:absolute;top:10px;right:14px;
      width:22px;height:22px;
      border-top:1px solid {border_clr};
      border-right:1px solid {border_clr};
      opacity:0.6;"></div>

  <!-- Label -->
  <div style="
      font-family:'Orbitron',monospace;
      font-size:1.5em;font-weight:900;
      color:{label_clr};
      letter-spacing:3px;text-transform:uppercase;
      text-shadow:0 0 20px {label_clr}66;
  ">{icon} {label_txt}</div>

  <!-- Probability bars -->
  <div style="margin-top:16px;">

    <!-- P(Hate) -->
    <div style="margin-bottom:10px;">
      <div style="display:flex;justify-content:space-between;align-items:center;
          margin-bottom:4px;">
        <span style="font-family:JetBrains Mono,monospace;font-size:0.72em;
            color:rgba(200,220,255,0.5);text-transform:uppercase;letter-spacing:1px;">
            P(Hate)
        </span>
        <span style="font-family:JetBrains Mono,monospace;font-size:0.82em;
            color:rgba(255,77,109,0.85);font-weight:500;">
            {prob_hate:.4f}
        </span>
      </div>
      <div style="background:rgba(255,255,255,0.07);border-radius:4px;height:14px;
          overflow:hidden;border:1px solid rgba(255,255,255,0.06);">
        <div style="background:{bar1_color};width:{hate_pct}%;height:100%;
            border-radius:4px;
            box-shadow:0 0 8px {bar1_color}66;
            transition:width 0.6s ease;"></div>
      </div>
    </div>

    <!-- P(Non-Hate) -->
    <div style="margin-bottom:8px;">
      <div style="display:flex;justify-content:space-between;align-items:center;
          margin-bottom:4px;">
        <span style="font-family:JetBrains Mono,monospace;font-size:0.72em;
            color:rgba(200,220,255,0.5);text-transform:uppercase;letter-spacing:1px;">
            P(Non-Hate)
        </span>
        <span style="font-family:JetBrains Mono,monospace;font-size:0.82em;
            color:rgba(0,255,136,0.75);font-weight:500;">
            {prob_nonhate:.4f}
        </span>
      </div>
      <div style="background:rgba(255,255,255,0.07);border-radius:4px;height:14px;
          overflow:hidden;border:1px solid rgba(255,255,255,0.06);">
        <div style="background:{bar2_color};width:{nonhate_pct}%;height:100%;
            border-radius:4px;
            box-shadow:0 0 8px {bar2_color}66;
            transition:width 0.6s ease;"></div>
      </div>
    </div>

    <div style="font-family:JetBrains Mono,monospace;font-size:0.7em;
        color:rgba(200,220,255,0.3);margin-top:4px;letter-spacing:0.5px;">
      θ (threshold) = {threshold:.4f}
    </div>
  </div>

  {spans_html}
</div>
"""