# Hate Speech Detector Demo

Demo Flask untuk deteksi ujaran kebencian dengan BLOOM-560M MTL, token labeling CRF, dan interpretabilitas LIME.

## Fitur

- Prediksi teks menjadi `HATE SPEECH` atau `NON-HATE`.
- Probabilitas hate/non-hate dan threshold kalibrasi.
- Highlight toxic span dan tabel BIO tag.
- LIME interpretability dengan progress bar saat proses perturbation berjalan.
- Visual LIME berbentuk bar kontribusi kata seperti dashboard referensi.
- Bubble contoh `language=mixed` dari dataset untuk demo cepat.
- Cache LIME untuk contoh demo agar hasil bisa muncul instan setelah pernah dihitung.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Pastikan model sudah tersedia di folder `models/`. Jika belum, jalankan:

```powershell
python download_models.py
```

## Menjalankan Aplikasi

```powershell
python app_flask.py
```

Default aplikasi berjalan di:

```text
http://127.0.0.1:7860
```

Port bisa diganti:

```powershell
python app_flask.py --port 5000
```

## Precompute LIME Contoh Demo

Sebelum presentasi, jalankan sekali:

```powershell
python app_flask.py --precompute-demo-lime
```

Perintah ini memuat model, menghitung LIME untuk semua bubble contoh `language=mixed`, lalu menyimpan cache di `models/lime_cache/`. Setelah cache tersedia, klik bubble contoh di UI akan menjalankan prediksi dan menampilkan LIME dari cache.

## Cara Pakai

1. Buka aplikasi di browser.
2. Masukkan teks pada input utama.
3. Tekan `Enter` atau tombol `ANALYZE`.
4. Untuk demo cepat, klik salah satu bubble contoh `CODE-MIXED DATASET EXAMPLES`.
5. Bubble contoh otomatis menjalankan prediksi dan LIME.
6. Untuk input manual, tekan `RUN LIME` setelah hasil prediksi muncul.
7. Progress bar akan berjalan selama LIME menghitung 200 perturbation samples, kecuali hasil sudah ada di cache.

## Catatan LIME

LIME di demo ini bersifat kualitatif. BLOOM menggunakan BPE/subword tokenizer, sedangkan LIME bekerja pada level kata, sehingga skor kontribusi dipakai sebagai indikasi interpretabilitas lokal, bukan bukti absolut.
