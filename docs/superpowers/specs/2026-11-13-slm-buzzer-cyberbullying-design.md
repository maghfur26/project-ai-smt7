# Analisis Komparatif Fine-Tuning Small Language Models (SLMs) untuk Deteksi Otomatis Perilaku Buzzer dan Cyberbullying di Media Sosial Instagram

Tanggal: 2026-11-13
Status: Disetujui (desain)
Penulis: Proyek mata kuliah Kecerdasan Buatan

## 1. Tujuan

Membangun dan membandingkan kinerja serta efisiensi sejumlah Small Language Model
(SLM) keluarga BERT yang di-fine-tune untuk dua tugas klasifikasi teks berbahasa
Indonesia:

1. Deteksi cyberbullying pada komentar media sosial.
2. Deteksi konten bergaya buzzer (buzzer-like content) menggunakan label lemah.

Hasil akhir berupa kode reproducible, artefak eksperimen (metrik, tabel, grafik),
laporan singkat, dan demo web Streamlit.

## 2. Ruang Lingkup

Termasuk:
- Pipeline akuisisi dan harmonisasi dataset publik.
- Prapemrosesan teks Indonesia (case folding, normalisasi slang, pembersihan noise).
- Pembentukan label lemah untuk tugas buzzer.
- Fine-tuning 5 model BERT pada 2 tugas dengan 3 seed.
- Evaluasi metrik klasifikasi dan efisiensi.
- Demo web lokal dengan Streamlit.
- Dokumentasi dataset, batasan, dan cara reproduksi.

Tidak termasuk:
- Scraping data Instagram secara langsung (kepatuhan ToS dan etika).
- Deteksi buzzer berbasis metadata akun sebagai label utama (hanya sebagai sinyal
  label lemah bila metadata tersedia).
- Deployment produksi, autentikasi, atau basis data.
- Hyperparameter search berskala besar.

## 3. Keputusan Desain

| Aspek | Keputusan |
|---|---|
| Dataset | Publik; utama Instagram, suplemen TikTok/Twitter |
| Tugas | Dua model klasifikasi terpisah |
| Model | `indobert-lite-base-p2`, `indobert-base-p1`, `indobertweet-base-uncased`, `bert-base-multilingual-cased`, `xlm-roberta-base` |
| Seed | 42, 123, 2024 |
| Komputasi | Google Colab / Kaggle GPU (T4/P100) |
| Demo | Streamlit lokal |
| Buzzer | Weak supervision, didokumentasikan sebagai proksi |

## 4. Arsitektur

Pipeline modular dengan pemisahan tanggung jawab:

```
configs -> src/data -> src/training -> results -> src/evaluation -> app
```

- `src/data`: unduh, harmonisasi label, cleaning, weak-label, split, dataset card.
- `src/models`: loader tokenizer/model berbasis nama konfigurasi.
- `src/training`: seeding, class weighting, HF Trainer, callback metrik.
- `src/evaluation`: metrik klasifikasi, metrik efisiensi, plot, error analysis.
- `scripts/`: entry point CLI yang tipis.
- `notebooks/`: EDA dan eksekusi Colab yang memanggil script.
- `app/`: Streamlit yang hanya memuat artefak hasil.

Semua parameter eksperimen berada di YAML. Tidak ada hyperparameter yang
di-hardcode di dalam modul inti.

## 5. Data

### 5.1 Cyberbullying

Sumber utama (diverifikasi pada implementasi):
- `cahkanor/id-cyberbullying-instagram` (300 komentar Instagram, seimbang).
- `rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia` (file komentar Instagram).
- Kaggle `cttrhnn/cyberbullying-bahasa-indonesia` (komentar Instagram).
- Kaggle `leonss0711/cleaned-indonesian-cyberbullying-dataset` (gabungan, MIT).
- `rhiosutoyo/Indonesian-TikTok-Cyberbullying-Comments-Dataset` (1.508, MIT).
- HuggingFace `SEACrowd/id_multilabel_hs` (suplemen hate speech, CC BY-NC-SA).

Label utama: biner `cyberbullying` / `non_cyberbullying`. Skema label per sumber
dipetakan ke skema biner; konflik label diselesaikan dengan aturan prioritas yang
dicatat di dataset card.

### 5.2 Buzzer (weak supervision)

Korpus masukan: kumpulan teks publik berbahasa Indonesia (utamakan yang memiliki
`user_id` dan `timestamp`). Label lemah dibentuk dari skor gabungan:

1. Near-duplicate: cosine similarity TF-IDF >= 0,85 dalam klaster teks.
2. Ukuran klaster near-duplicate >= ambang minimum.
3. Kepadatan hashtag/mention/tautan melebihi ambang.
4. Frekuensi posting tinggi per akun dalam jendela waktu.
5. Kecocokan leksikon promosi/agitasi.

Teks dengan skor >= ambang diberi label `buzzer_like`; teks unik dan bersih diberi
`non_buzzer`. Ambang disimpan di konfigurasi dan dilaporkan.

Batasan eksplisit: label ini adalah proksi berbasis teks, bukan verifikasi
perilaku buzzer sebenarnya. Klaim pada laporan dibatasi pada "konten bergaya
buzzer".

### 5.3 Prapemrosesan dan Split

- Case folding dan normalisasi singkatan/slang Indonesia.
- Penghapusan URL, mention, hashtag (opsi mempertahankan emoji).
- Deduplikasi teks dan near-duplicate sebelum split.
- Split stratified 80/10/10.
- Group split berdasarkan `user_id` atau klaster near-duplicate untuk mencegah
  kebocoran.
- Class weighting untuk menangani ketidakseimbangan.

## 6. Pelatihan

- `max_length=128`, `batch_size` 16-32, `learning_rate=2e-5`, `epochs` 3-5,
  `warmup_ratio=0.1`, `weight_decay=0.01`.
- Early stopping pada validation macro-F1, patience 2.
- FP16 jika GPU mendukung; gradient clipping 1.0.
- Loss: cross-entropy dengan class weights.
- Setiap run menyimpan: checkpoint terbaik, `metrics.json`, `trainer_state`,
  waktu training, jumlah parameter, ukuran model, latency inferensi.
- Checkpoint dan log disimpan ke Google Drive saat berjalan di Colab.

Jumlah run: 5 model x 2 tugas x 3 seed = 30 run.

## 7. Evaluasi

Metrik klasifikasi:
- Accuracy
- Macro dan weighted precision, recall, F1
- ROC-AUC (biner)
- Confusion matrix dan classification report per kelas

Metrik efisiensi:
- Jumlah parameter
- Ukuran model (MB)
- Waktu training
- Latency inferensi per sampel (batch 1 dan 32)
- Throughput
- Peak GPU memory (jika tersedia)

Agregasi: mean ± std antar-seed. Perbandingan divisualisasikan sebagai bar chart,
box plot, dan heatmap. Analisis error mencakup contoh salah klasifikasi dan
perbedaan domain antar-sumber. Uji Wilcoxon opsional dengan catatan keterbatasan
jumlah seed.

## 8. Demo Web

Streamlit satu halaman:
- Pemilihan tugas (cyberbullying/buzzer).
- Pemilihan model (default model terbaik).
- Input teks bebas dan tombol analisis.
- Menampilkan label, confidence, dan distribusi probabilitas.
- Tab perbandingan berisi tabel metrik dan grafik dari `results/`.
- Contoh teks preset.

Model dimuat dari `results/models/<run_id>/`. Jika artefak belum ada, aplikasi
menampilkan instruksi, bukan gagal diam-diam.

## 9. Pengujian

- Unit test: harmonisasi label, cleaning, aturan weak-label, split tanpa leakage.
- Smoke test: inferensi dengan model dummy/forward pass kecil.
- Verifikasi end-to-end: satu run baseline pada subset kecil.
- `pytest` sebagai perintah utama.

## 10. Risiko dan Mitigasi

| Risiko | Mitigasi |
|---|---|
| Dataset buzzer tidak tersedia atau lisensi tidak jelas | Weak-label + dokumentasi batasan; opsi fokus cyberbullying |
| Kebocoran near-duplicate | Dedup + group split |
| Domain mismatch Instagram vs Twitter/TikTok | Laporkan hasil per sumber, bahas sebagai limitation |
| Kuota Colab habis | Checkpoint ke Drive, run per model, fallback Kaggle |
| Class imbalance | Class-weighted loss, macro-F1 sebagai metrik utama |
| Hasil tidak stabil | 3 seed dan mean ± std |

## 11. Kriteria Penerimaan

1. `python scripts/prepare_data.py --task <task>` menghasilkan dataset processed
   dan split yang tervalidasi.
2. `python scripts/run_experiment.py --config <yaml>` menyelesaikan satu run dan
   menulis metrik.
3. `python scripts/collect_results.py` menghasilkan `metrics.csv` dan tabel agregat.
4. `streamlit run app/streamlit_app.py` menampilkan prediksi untuk kedua tugas.
5. `pytest` lulus.
6. README menjelaskan setup, sumber data, cara menjalankan, dan batasan.

## 12. Struktur Repo

```
configs/
data/{raw,interim,processed}
src/{data,models,training,evaluation,utils}
scripts/
notebooks/
results/{metrics.csv,tables,figures,models,logs}
app/
tests/
README.md
requirements.txt
```
