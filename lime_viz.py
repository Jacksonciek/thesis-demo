"""
lime_viz.py — LIME Interpretability Analysis
=============================================
Mereplikasi lime_predict_proba_v4() dan explain_prediction_v4() dari NB05v2.

Penting (dicatat di NB05v2 bagian Keterbatasan LIME):
  1. BLOOM memakai BPE tokenizer; LIME beroperasi di level kata → mismatch
     subword. Kontribusi kata bersifat APPROKSIMASI.
  2. LIME adalah pendekatan lokal-linear → hanya valid di sekitar prediksi.
  3. Gunakan sebagai analisis KUALITATIF, bukan kuantitatif.

Jumlah sampel dikurangi dari 1000 (NB05) menjadi 200 (CPU-friendly).
Chart menggunakan dark theme agar cocok dengan glassmorphism UI.
"""

import re
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")               # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from lime.lime_text import LimeTextExplainer
import torch
from PIL import Image


# ── Dark theme constants ────────────────────────────────────────────────
_BG_DARK    = "#050a18"
_BG_AXES    = "#0a1020"
_TEXT_COLOR = "#dceeff"
_TEXT_DIM   = "#6a8aaa"
_GRID_COLOR = "#1a2a3a"
_NEON_RED   = "#ff4d6d"
_NEON_BLUE  = "#00b4ff"
_NEON_GREEN = "#00ff88"
_NEON_AMB   = "#ffaa00"


def _apply_dark_style():
    """Apply dark futuristic style to all subsequent matplotlib figures."""
    rcParams.update({
        "figure.facecolor":  _BG_DARK,
        "axes.facecolor":    _BG_AXES,
        "axes.edgecolor":    _GRID_COLOR,
        "axes.labelcolor":   _TEXT_DIM,
        "xtick.color":       _TEXT_DIM,
        "ytick.color":       _TEXT_COLOR,
        "text.color":        _TEXT_COLOR,
        "grid.color":        _GRID_COLOR,
        "grid.alpha":        1.0,
        "legend.facecolor":  "#0d1828",
        "legend.edgecolor":  "#1a2a3a",
        "legend.labelcolor": _TEXT_DIM,
        "font.family":       "monospace",
        "font.size":         10,
    })


# ══════════════════════════════════════════════════════════════════════════
#  LIME predict_proba — identik dengan lime_predict_proba_v4() di NB05v2
# ══════════════════════════════════════════════════════════════════════════
def _make_predict_fn(tokenizer, model, device, max_length: int = 512):
    """
    Mengembalikan fungsi predict_proba yang diterima oleh LimeTextExplainer.
    Fungsi ini:
      • Menerima list[str] (perturbed texts dari LIME)
      • Mengembalikan np.ndarray shape (N, 2) [P(NonHate), P(Hate)]
    Alur identik dengan lime_predict_proba_v4() di NB05v2.
    """
    BATCH_SIZE = 8   # lebih kecil agar hemat RAM CPU

    def predict_proba(texts: list) -> np.ndarray:
        model.eval()
        all_probs = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = [
                re.sub(r"\s+", " ", str(t)).strip() or "."
                for t in texts[i : i + BATCH_SIZE]
            ]
            enc = tokenizer(
                batch_texts,
                padding        = "max_length",
                truncation     = True,
                max_length     = max_length,
                return_tensors = "pt",
            )
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids, attention_mask)
            temperature = float(getattr(model, "temperature", 1.0))
            probs = torch.softmax(
                out["sentence_logits"].float() / temperature, dim=-1
            )
            all_probs.append(probs.cpu().numpy())
        return np.vstack(all_probs)   # (N, 2)

    return predict_proba


# ══════════════════════════════════════════════════════════════════════════
#  Fungsi eksplanasi utama
# ══════════════════════════════════════════════════════════════════════════
def run_lime_explanation(
    text:         str,
    tokenizer,
    model,
    device:       torch.device,
    actual_label: int,          # 0=NonHate, 1=Hate  (dipakai sbg explain_label)
    num_features: int  = 10,
    num_samples:  int  = 200,   # 200 untuk CPU (NB05 pakai 1000)
    max_length:   int  = 512,
) -> tuple:
    """
    Menjalankan LIME dan mengembalikan:
      (matplotlib_figure, list_of_word_contributions, pred_label, confidence)

    Parameters
    ----------
    actual_label : int
        Label yang digunakan LIME untuk menjelaskan (0 atau 1).
        Di NB05v2: explain_label = actual_label saat label diketahui.
        Di demo, kita pakai hasil prediksi model sebagai proxy.
    num_samples  : int
        Jumlah perturbasi LIME. Lebih banyak = lebih akurat tapi lebih lambat.
        200 samples ≈ ~30–90 detik di CPU.

    Returns
    -------
    fig          : matplotlib.figure.Figure  (untuk gr.Plot)
    contributions: list[tuple[str, float]]
    pred_label   : str
    confidence   : float
    """
    text_norm = re.sub(r"\s+", " ", str(text)).strip()
    if not text_norm:
        return None, [], "—", 0.0

    predict_fn = _make_predict_fn(tokenizer, model, device, max_length)

    # Validasi: ambil prediksi model untuk teks ini
    probs      = predict_fn([text_norm])[0]   # [P(NH), P(H)]
    pred_idx   = int(probs.argmax())
    pred_label = "HATE SPEECH" if pred_idx == 1 else "NON-HATE"
    confidence = float(probs[pred_idx])

    # explain_label: ikuti actual_label jika tersedia, else pred
    explain_label = actual_label

    # ── Inisialisasi LIME (identik NB05v2) ─────────────────────────────
    explainer = LimeTextExplainer(
        class_names  = ["Non-Hate", "Hate"],
        random_state = 42,
    )

    exp = explainer.explain_instance(
        text_norm,
        predict_fn,
        num_features = num_features,
        num_samples  = num_samples,
        labels       = [0, 1],
    )

    contributions = exp.as_list(label=explain_label)

    # ── Buat chart ──────────────────────────────────────────────────────
    fig = _build_lime_chart(
        contributions = contributions,
        explain_label = explain_label,
        pred_label    = pred_label,
        confidence    = confidence,
        text_norm     = text_norm,
        num_samples   = num_samples,
    )

    return fig, contributions, pred_label, confidence


# ══════════════════════════════════════════════════════════════════════════
#  Chart builder — horizontal bar chart (dark theme)
# ══════════════════════════════════════════════════════════════════════════
def _build_lime_chart(
    contributions: list,
    explain_label: int,
    pred_label:    str,
    confidence:    float,
    text_norm:     str,
    num_samples:   int,
) -> plt.Figure:
    """
    Membuat dark-theme LIME chart:
      • Batang merah neon = kontribusi ke arah HATE
      • Batang biru neon  = kontribusi ke arah NON-HATE
    Diurutkan dari kontribusi terbesar (atas) ke terkecil (bawah).
    """
    _apply_dark_style()

    if not contributions:
        fig, ax = plt.subplots(figsize=(8, 3), facecolor=_BG_DARK)
        ax.set_facecolor(_BG_AXES)
        ax.text(0.5, 0.5, "Tidak ada kontribusi tersedia.",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=12,
                color=_TEXT_DIM)
        ax.axis("off")
        return fig

    # Sort by absolute value
    sorted_contribs = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
    words   = [item[0] for item in sorted_contribs]
    weights = [item[1] for item in sorted_contribs]

    label_name = "Hate" if explain_label == 1 else "Non-Hate"

    # ── Color logic ─────────────────────────────────────────────────────
    if explain_label == 1:
        # Explaining HATE: positive = supports hate (red), negative = against (blue)
        bar_colors = [
            (_NEON_RED  if w > 0 else _NEON_BLUE)
            for w in weights
        ]
        bar_alphas = [
            min(1.0, 0.5 + abs(w) * 8)
            for w in weights
        ]
        alpha_colors = [
            f"{c}{int(a*255):02x}"
            for c, a in zip(bar_colors, bar_alphas)
        ]
    else:
        # Explaining NON-HATE: positive = supports non-hate (green), negative = hate (red)
        bar_colors = [
            (_NEON_GREEN if w > 0 else _NEON_RED)
            for w in weights
        ]
        bar_alphas = [
            min(1.0, 0.5 + abs(w) * 8)
            for w in weights
        ]
        alpha_colors = [
            f"{c}{int(a*255):02x}"
            for c, a in zip(bar_colors, bar_alphas)
        ]

    n     = len(words)
    fig_h = max(5, n * 0.5 + 3)
    fig, ax = plt.subplots(figsize=(10, fig_h), facecolor=_BG_DARK)
    ax.set_facecolor(_BG_AXES)

    y_pos = list(range(n - 1, -1, -1))   # terbalik agar terbesar di atas
    bars  = ax.barh(
        y_pos, weights,
        color     = bar_colors,
        alpha     = 0.85,
        edgecolor = "none",
        height    = 0.6,
    )

    # Add glow effect via a second semi-transparent wider bar
    for bar, bc in zip(bars, bar_colors):
        ax.barh(
            bar.get_y() + bar.get_height() / 2,
            bar.get_width(),
            height    = 0.75,
            color     = bc,
            alpha     = 0.15,
            edgecolor = "none",
        )

    # Value labels at bar ends
    for bar, w in zip(bars, weights):
        x_off  = 0.0008 if w >= 0 else -0.0008
        ha     = "left" if w >= 0 else "right"
        color  = _NEON_RED if w > 0 and explain_label == 1 else (
                 _NEON_GREEN if w > 0 and explain_label == 0 else _NEON_BLUE)
        ax.text(
            w + x_off,
            bar.get_y() + bar.get_height() / 2,
            f"{w:+.4f}",
            va="center", ha=ha,
            fontsize=8.5, fontweight="bold",
            color=color,
            fontfamily="monospace",
        )

    # Y-axis (word labels)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        words, fontsize=10,
        color=_TEXT_COLOR,
        fontfamily="monospace",
    )

    # Zero line
    ax.axvline(0, color=_GRID_COLOR, linewidth=1.2, alpha=0.8)

    # X-axis label
    ax.set_xlabel(
        f"Kontribusi kata terhadap label '{label_name}'",
        fontsize=10, color=_TEXT_DIM,
        fontfamily="monospace",
        labelpad=10,
    )

    # Grid
    ax.grid(axis="x", color=_GRID_COLOR, linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    # Spines
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID_COLOR)
        spine.set_linewidth(0.8)

    # Legend
    if explain_label == 1:
        patch_pos = mpatches.Patch(color=_NEON_RED,   label="→ Mendukung HATE",     alpha=0.85)
        patch_neg = mpatches.Patch(color=_NEON_BLUE,  label="→ Mendukung NON-HATE", alpha=0.85)
    else:
        patch_pos = mpatches.Patch(color=_NEON_GREEN, label="→ Mendukung NON-HATE", alpha=0.85)
        patch_neg = mpatches.Patch(color=_NEON_RED,   label="→ Mendukung HATE",     alpha=0.85)

    legend = ax.legend(
        handles  = [patch_pos, patch_neg],
        fontsize = 8.5,
        loc      = "lower right",
        framealpha = 0.6,
    )
    for text in legend.get_texts():
        text.set_color(_TEXT_DIM)

    # Title
    is_hate_pred = pred_label == "HATE SPEECH"
    title_color  = _NEON_RED if is_hate_pred else _NEON_GREEN
    ax.set_title(
        f"LIME WORD CONTRIBUTIONS\n"
        f"Model: {pred_label}  ({confidence:.2%} confidence)   "
        f"| Label Explained: '{label_name}'   | Samples: {num_samples}",
        fontsize      = 9.5,
        color         = title_color,
        pad           = 14,
        fontfamily    = "monospace",
        fontweight    = "bold",
    )

    # Footer warning
    fig.text(
        0.5, -0.015,
        "⚠  LIME bersifat KUALITATIF: BPE tokenizer BLOOM ≠ word-level LIME. "
        "Gunakan sebagai indikasi, bukan kebenaran absolut.",
        ha        = "center",
        fontsize  = 7.5,
        color     = "#3a5a7a",
        style     = "italic",
        transform = fig.transFigure,
        fontfamily = "monospace",
    )

    plt.tight_layout(pad=1.8)
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  Helper: figure → PIL Image (untuk gr.Image di Gradio)
# ══════════════════════════════════════════════════════════════════════════
def fig_to_pil(fig: plt.Figure) -> Image.Image:
    """Mengkonversi matplotlib figure ke PIL Image untuk Gradio."""
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png",
        dpi=130,
        bbox_inches="tight",
        facecolor=_BG_DARK,
        edgecolor="none",
    )
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img
