"""
model_utils.py — Arsitektur BloomForMTL_v2 & loader
=====================================================
Kelas BloomForMTL_v2 diambil VERBATIM dari NB05v2 (05_evaluation_lime_v2)
agar state_dict dari checkpoint berhasil dimuat tanpa missing/unexpected keys.

Fungsi load_model() mereplikasi load_model() dari NB05v2 secara identik.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BloomPreTrainedModel,
    BloomModel,
    BloomConfig,
    BloomTokenizerFast,
    AutoTokenizer,
)
from torchcrf import CRF

# ── Konstanta ─────────────────────────────────────────────────────────────
IGNORE_IDX = -100
LABEL2ID   = {"O": 0, "B-TOX": 1, "I-TOX": 2}
ID2LABEL   = {0: "O", 1: "B-TOX", 2: "I-TOX"}


# ══════════════════════════════════════════════════════════════════════════
#  Arsitektur UTAMA — identik dengan NB05v2
# ══════════════════════════════════════════════════════════════════════════
class BloomForMTL_v2(BloomPreTrainedModel):
    """
    BLOOM-560m dengan dua head:
      • Sentence head : pooled mean → Linear(H, 2)   (hate/non-hate)
      • Token head    : per-token  → Linear(H, 3) + CRF  (O/B-TOX/I-TOX)

    Parameter alpha, beta, gamma, label_smoothing, use_crf, temperature
    dibaca dari checkpoint saat loading — tidak perlu diset manual.
    """

    def __init__(
        self,
        config,
        num_sentence_labels: int   = 2,
        num_token_labels:    int   = 3,
        dropout:             float = 0.1,
        alpha:               float = 0.5,
        beta:                float = 0.5,
        label_smoothing:     float = 0.1,
        use_crf:             bool  = True,
        gamma:               float = 0.0,
    ):
        super().__init__(config)
        self.bloom            = BloomModel(config)
        self.alpha            = alpha
        self.beta             = beta
        self.label_smoothing  = label_smoothing
        self.use_crf          = use_crf
        self.gamma            = gamma
        self._token_class_weights = None          # hanya dipakai saat training
        self.dropout          = nn.Dropout(dropout)

        h = config.hidden_size
        self.sentence_classifier = nn.Linear(h, num_sentence_labels)
        self.token_classifier    = nn.Linear(h, num_token_labels)

        if self.use_crf:
            self.crf = CRF(num_token_labels, batch_first=True)

        self.post_init()

    # Diperlukan agar transformers >= 4.44 tidak crash saat class tidak
    # didefinisikan di dalam file Python (bukan notebook).
    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        return False

    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Inference-only forward pass.
        Tidak menghitung loss (sentence_labels / token_labels diabaikan).
        """
        out    = self.bloom(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state                          # (B, T, H)

        # ── Sentence Head: masked mean pooling ──────────────────────────
        mask_e = attention_mask.unsqueeze(-1).float()
        pooled = self.dropout(
            (hidden * mask_e).sum(1) / mask_e.sum(1).clamp(min=1e-9)
        )
        sentence_logits = self.sentence_classifier(pooled)     # (B, 2)

        # ── Token Head ───────────────────────────────────────────────────
        token_emissions = self.token_classifier(self.dropout(hidden))  # (B, T, 3)

        if self.use_crf:
            # Viterbi decoding — returns list of lists (variable length)
            token_preds_viterbi = self.crf.decode(
                token_emissions.float(),
                mask=attention_mask.bool(),
            )
        else:
            # use_crf=False (Baseline 1): argmax over class dim (-1 = dim 2 of B×T×3)
            # Only return predictions for non-padded positions (mask==1),
            # matching CRF decode behavior (variable-length sublists).
            raw_argmax = token_emissions.argmax(-1).cpu().tolist()  # (B, T)
            token_preds_viterbi = [
                [lbl for lbl, m in zip(seq, mask_seq) if m == 1]
                for seq, mask_seq in zip(raw_argmax, attention_mask.cpu().tolist())
            ]

        return {
            "sentence_logits": sentence_logits,
            "token_logits":    token_emissions,
            "token_preds_crf": token_preds_viterbi,
        }


# ══════════════════════════════════════════════════════════════════════════
#  Model Loader — identik dengan load_model() di NB05v2
# ══════════════════════════════════════════════════════════════════════════
def load_model(
    checkpoint_path: str,
    bloom_local_path: str,
    device: torch.device,
) -> BloomForMTL_v2:
    """
    Memuat checkpoint main_bloom_mtl_v2/best_model_hm.pt ke objek BloomForMTL_v2.

    Alur identik dengan load_model() di NB05v2:
      1. torch.load checkpoint dict
      2. Baca hyperparameter dari dict (alpha, beta, dropout, temperature, dst.)
      3. Bangun model dengan BloomConfig dari lokal cache
      4. load_state_dict(strict=False)
      5. Validasi critical keys
      6. model.eval()

    Returns
    -------
    BloomForMTL_v2 dalam eval mode dengan atribut .temperature diset.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {checkpoint_path}\n"
            f"Pastikan file sudah diunduh dari Google Drive."
        )

    print(f"[Model] Memuat checkpoint dari: {checkpoint_path}")
    print(f"[Model] Ini mungkin memakan waktu 30–120 detik (file ~4.2 GB)...")

    # ── Step 1: Load raw checkpoint ────────────────────────────────────
    raw_ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,       # diperlukan karena checkpoint berisi dict non-tensor
    )

    # ── Step 2: Ekstrak hyperparameter (persis seperti NB05v2) ─────────
    if isinstance(raw_ckpt, dict):
        dropout        = raw_ckpt.get("dropout",         0.1)
        alpha          = raw_ckpt.get("alpha",           0.5)
        beta           = raw_ckpt.get("beta",            0.5)
        label_smooth   = raw_ckpt.get("label_smoothing", 0.1)
        use_crf        = raw_ckpt.get("use_crf",         True)
        gamma_loaded   = raw_ckpt.get("gamma",           0.0)
        state          = raw_ckpt.get("model_state",     raw_ckpt)
        epoch          = raw_ckpt.get("epoch",           "?")
        temperature    = raw_ckpt.get("temperature",     1.0)
    else:
        state          = raw_ckpt
        epoch          = "?"
        dropout        = 0.1
        alpha          = 0.5
        beta           = 0.5
        label_smooth   = 0.1
        use_crf        = True
        gamma_loaded   = 0.0
        temperature    = 1.0

    # Bebaskan memori checkpoint mentah secepatnya
    del raw_ckpt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Step 3: Bangun model ────────────────────────────────────────────
    config = BloomConfig.from_pretrained(bloom_local_path)
    model  = BloomForMTL_v2(
        config,
        num_sentence_labels = 2,
        num_token_labels    = 3,
        dropout             = dropout,
        alpha               = alpha,
        beta                = beta,
        label_smoothing     = label_smooth,
        use_crf             = use_crf,
        gamma               = gamma_loaded,
    ).to(device)

    # ── Step 4: Load state dict ─────────────────────────────────────────
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
    del state

    # ── Step 5: Validasi kritis ─────────────────────────────────────────
    critical_missing = [k for k in missing_keys if "bloom." in k]
    if critical_missing:
        raise RuntimeError(
            f"FATAL: BLOOM backbone weights hilang dari checkpoint "
            f"({len(critical_missing)} keys). Contoh: {critical_missing[:3]}"
        )

    if use_crf:
        crf_missing = [k for k in missing_keys if "crf." in k]
        if crf_missing:
            print(
                f"[WARNING] CRF weights tidak ada di checkpoint ({len(crf_missing)} keys). "
                f"CRF diinisialisasi secara acak — pastikan checkpoint v2 benar."
            )

    if missing_keys:
        print(f"[INFO] Non-critical missing keys ({len(missing_keys)}): {missing_keys[:3]}")
    if unexpected_keys:
        print(f"[INFO] Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:3]}")

    # ── Step 6: Set eval mode + temperature ────────────────────────────
    model.eval()
    model.temperature = float(temperature)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"[Model] ✅ Dimuat: epoch={epoch}, α={alpha:.1f}, β={beta:.1f}, "
        f"T={temperature:.4f}, use_crf={use_crf}, γ={gamma_loaded}"
    )
    return model


# ══════════════════════════════════════════════════════════════════════════
#  Tokenizer Loader
# ══════════════════════════════════════════════════════════════════════════
def load_tokenizer(bloom_local_path: str) -> BloomTokenizerFast:
    """
    Memuat BloomTokenizerFast dari lokal cache dengan patch tokenizer_class
    (identik dengan NB03v2/NB05v2).
    """
    tok_cfg_path = os.path.join(bloom_local_path, "tokenizer_config.json")
    tok_json     = os.path.join(bloom_local_path, "tokenizer.json")

    # Patch tokenizer_class jika corrupt (sama dengan NB03/NB05)
    if os.path.exists(tok_cfg_path):
        with open(tok_cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("tokenizer_class") in {"TokenizersBackend", None, ""}:
            cfg["tokenizer_class"] = "BloomTokenizerFast"
            with open(tok_cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            print("[Tokenizer] tokenizer_config.json di-patch → BloomTokenizerFast")

    if os.path.exists(tok_json):
        try:
            tokenizer = BloomTokenizerFast.from_pretrained(bloom_local_path)
            print(f"[Tokenizer] ✅ Dimuat dari lokal: {bloom_local_path}")
        except Exception as e:
            print(f"[Tokenizer] ⚠️ Fallback ke HuggingFace Hub ({e})...")
            tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
            tokenizer.save_pretrained(bloom_local_path)
            print("[Tokenizer] ✅ Dimuat dari HuggingFace dan disimpan lokal.")
    else:
        tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
        tokenizer.save_pretrained(bloom_local_path)
        print("[Tokenizer] ✅ Dimuat dari HuggingFace Hub (lokal tidak ada).")

    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


# ══════════════════════════════════════════════════════════════════════════
#  Threshold Loader
# ══════════════════════════════════════════════════════════════════════════
def load_threshold(threshold_json_path: str, fallback: float = 0.4429) -> float:
    """
    Membaca best_threshold_v2 dari threshold_info_v2.json.
    Jika file tidak ada, gunakan nilai fallback dari NB05v2 (0.4429).
    """
    if os.path.exists(threshold_json_path):
        with open(threshold_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        thr = float(data.get("best_threshold_v2", fallback))
        print(f"[Threshold] ✅ Dimuat dari JSON: {thr:.4f}")
        return thr
    else:
        print(f"[Threshold] ⚠️ File tidak ditemukan, pakai fallback: {fallback:.4f}")
        return fallback