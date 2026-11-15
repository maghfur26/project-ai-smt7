# Analisis Komparatif Fine-Tuning SLM untuk Deteksi Buzzer dan Cyberbullying Instagram

Proyek mata kuliah Kecerdasan Buatan. Repositori ini membandingkan lima Small
Language Model (SLM) keluarga BERT yang di-fine-tune untuk dua tugas berbahasa
Indonesia:

1. **Cyberbullying** — klasifikasi biner komentar media sosial.
2. **Buzzer-like content** — klasifikasi biner konten bergaya buzzer menggunakan
   label lemah (weak supervision).

Keluaran: pipeline reproducible, metrik klasifikasi + efisiensi, tabel/grafik
perbandingan, dan demo web Streamlit.

> **Catatan metodologis penting.** Dataset publik berlabel "buzzer" di level teks
> tidak tersedia. Tugas buzzer di sini memakai **label proksi** dari aturan
> repetisi/kemiripan teks, kepadatan hashtag/tautan, dan leksikon promosi/agitasi.
> Hasilnya bukan verifikasi perilaku buzzer sebenarnya. Lihat
> [`data/DATASET_CARD.md`](data/DATASET_CARD.md).

## Ringkasan Keputusan

| Aspek | Nilai |
|---|---|
| Model | IndoBERT-lite, IndoBERT-base, IndoBERTweet, mBERT, XLM-R base (+ DistilBERT opsional) |
| Seed | 42, 123, 2024 |
| Data cyberbullying | 2.208 baris mentah; 2.191 setelah dedup & pembersihan konflik label |
| Data buzzer | 900 tweet politik; 890 setelah pembersihan; 120 label lemah positif |
| Split | 80/10/10, group-aware (anti-kebocoran near-duplicate) |
| Metrik utama | Macro-F1 |
| Efisiensi | Parameter, ukuran model, waktu training, latency inferensi |

## Struktur Repo

```
configs/                  # data + eksperimen YAML
data/
  raw/                    # unduhan mentah (tidak di-commit)
  processed/              # train/val/test + stats.json
  DATASET_CARD.md
src/
  data/                   # unduh, cleaning, label, weak-label, split, tokenisasi
  models/                 # loader + inferensi
  training/               # metrik + trainer
  evaluation/             # agregasi + plot
  utils/                  # config, IO, seeding
scripts/                  # entry point CLI
notebooks/                # EDA + alur Colab
app/streamlit_app.py      # demo web
tests/                    # pytest
results/                  # metrics.csv, tables, figures, models
```

## Setup

Python 3.10 atau 3.11 (Colab memakai 3.10+). GPU sangat disarankan untuk
fine-tuning, tetapi pipeline data dan unit test berjalan di CPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Untuk verifikasi cepat tanpa GPU:

```bash
pytest -q
```

## 1. Siapkan Dataset

```bash
python scripts/prepare_data.py --config configs/data/cyberbullying.yaml
python scripts/prepare_data.py --config configs/data/buzzer.yaml
```

Perintah ini mengunduh sumber GitHub publik, membersihkan teks, membentuk label,
membuat split anti-kebocoran, dan menulis `stats.json`.

Sumber Kaggle tidak diunduh otomatis karena memerlukan autentikasi. Tempatkan file
secara manual di `data/raw/` lalu tambahkan entri `path` pada YAML. Lihat
`data/DATASET_CARD.md`.

## 2. Fine-Tuning

Satu run:

```bash
python scripts/run_experiment.py --config configs/cyberbullying/indobert-lite.yaml
```

Semua model x semua seed untuk satu tugas:

```bash
python scripts/run_all.py --task cyberbullying
python scripts/run_all.py --task buzzer
```

Filter model/seed tertentu:

```bash
python scripts/run_all.py --task cyberbullying --models indobert-lite,mbert --seeds 42
python scripts/run_all.py --task buzzer --include-optional
python scripts/run_all.py --task cyberbullying --dry-run
```

Untuk menyimpan metrik tiap run ke lokasi lain (misalnya Google Drive di Colab)
agar aman saat runtime ter-reset:

```bash
python scripts/run_all.py --task cyberbullying --backup-dir /content/drive/MyDrive/slm_metrics
```

Checkpoint trainer dihapus otomatis setelah model final disimpan, sehingga 30 run
tidak memenuhi disk Colab.

Artefak per run disimpan di `results/models/<run_name>/`:

```
metrics.json          # akurasi, macro/weighted P/R/F1, ROC-AUC
efficiency.json       # parameter, ukuran, waktu, latency, GPU memory
test_report.json      # confusion matrix + classification report
run_summary.json      # konfigurasi + metrik training/test
model/                # checkpoint + tokenizer
```

## 3. Agregasi dan Grafik

```bash
python scripts/collect_results.py
```

Menghasilkan:

- `results/metrics.csv` — satu baris per run
- `results/tables/aggregate.csv` — mean ± std per model/tugas
- `results/figures/macro_f1_<task>.png`
- `results/figures/efficiency_<task>.png`
- `results/figures/macro_f1_by_seed.png`

## 4. Demo Web

```bash
streamlit run app/streamlit_app.py
```

Fitur: pilih tugas dan model, masukkan teks, lihat label + confidence + distribusi
probabilitas, serta tab perbandingan metrik/grafik.

## 5. Prediksi via CLI

```bash
python scripts/predict.py \
  --model-dir results/models/cyber_indobert-lite_s42/model \
  --task cyberbullying \
  --text "Dasar bodoh, muka lo jelek banget"
```

## Colab / Kaggle

Repositori: https://github.com/maghfur26/project-ai-smt7

Buka `notebooks/02_colab_train.ipynb` di Colab. Notebook melakukan:

0. Clone repo ke `/content/project-ai-smt7`.
1. Pemeriksaan GPU.
2. Instalasi dependency.
3. Mount Google Drive untuk checkpoint.
4. Prepare data.
5. Training semua model/seed.
6. Agregasi hasil.
7. Unduh artefak.

Alternatif jika ingin memakai arsip: unggah `slm-buzzer-cyberbullying.zip`
(tanpa `.venv`, data mentah, dan checkpoint) ke panel **Files**, ekstrak ke
`/content/`, lalu sesuaikan `PROJECT_DIR`.

Estimasi: 5 model x 2 tugas x 3 seed = 30 run. Pada dataset kecil, satu run
IndoBERT-lite di GPU T4 berkisar beberapa menit; total 5–8 jam GPU. Untuk kuota
terbatas, jalankan per model/seed dan lanjutkan dari checkpoint.

## Hasil yang Diharapkan

Tabel perbandingan berisi, per model dan tugas:

- Accuracy, macro/weighted precision/recall/F1, ROC-AUC.
- Jumlah parameter, ukuran model, waktu training, latency per sampel.
- Rata-rata dan standar deviasi antar-seed.

Analisis error dan perbedaan domain (Instagram vs TikTok/Twitter) dibahas di
laporan, bukan dihasilkan otomatis.

## Batasan dan Etika

- Tidak melakukan scraping Instagram; seluruh data berasal dari dataset publik.
- Data dapat mengandung ujaran kebencian; hanya untuk penelitian.
- Label buzzer adalah proksi lemah, bukan ground truth perilaku.
- Model tidak boleh dipakai untuk moderasi nyata tanpa peninjauan manusia.
- Domain data didominasi Twitter/TikTok; generalisasi ke Instagram adalah batasan.
- Dataset dapat memiliki lisensi yang berbeda; lihat dataset card.

## Status Verifikasi

- `pytest -q`: **55 passed**.
- Prepare data publik cyberbullying: 2.208 baris mentah, 2.191 setelah dedup
  (7 baris dengan label bertentangan dibuang), split 1.751/221/219.
- Prepare data buzzer: 900 baris, 890 setelah pembersihan, 120 label lemah
  positif, split 711/88/91.
- Smoke fine-tuning IndoBERT-lite di CPU (subset 240 baris, 1 epoch):
  berhasil menulis `metrics.json`, `efficiency.json`, `test_report.json`,
  `run_summary.json`, dan `model/`.
- Smoke inferensi CLI dan `collect_results.py`: berhasil.

Catatan: angka smoke test bukan hasil penelitian; jalankan eksperimen penuh di GPU.

## Referensi Data

- `cahkanor/id-cyberbullying-instagram` (Instagram, 300 komentar)
- `rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia` (Instagram, 400 komentar)
- `rhiosutoyo/Indonesian-TikTok-Cyberbullying-Comments-Dataset` (TikTok, 1.508 komentar, MIT)
- `leonss0711/cleaned-indonesian-cyberbullying-dataset` (Kaggle, gabungan, MIT)
- `SEACrowd/id_multilabel_hs` (Hugging Face, 13.169 tweet, CC BY-NC-SA 4.0)
- `rizalespe/.../dataset_tweet_sentiment_pilkada_DKI_2017.csv` (korpus tweet politik)
