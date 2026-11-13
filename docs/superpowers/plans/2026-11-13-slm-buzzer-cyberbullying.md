# SLM Buzzer & Cyberbullying Implementation Plan

> **Status:** Implemented (2026-11-13). Verification details are in `README.md`
> and all 51 tests pass.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Membangun pipeline reproducible untuk fine-tuning dan perbandingan 5 SLM BERT pada deteksi cyberbullying dan konten buzzer berbahasa Indonesia, lengkap dengan evaluasi dan demo Streamlit.

**Architecture:** Package Python modular (`src/`) dengan konfigurasi YAML, entry point CLI tipis (`scripts/`), artefak eksperimen di `results/`, dan demo Streamlit yang hanya membaca artefak. Data diproses melalui tahap unduh -> harmonisasi label -> cleaning -> dedup -> split -> tokenisasi.

**Tech Stack:** Python 3.10/3.11, pandas, scikit-learn, PyTorch, transformers, datasets, evaluate, streamlit, pytest, PyYAML.

**Spec:** `docs/superpowers/specs/2026-11-13-slm-buzzer-cyberbullying-design.md`

## Global Constraints

- Bahasa teks: Indonesia. Nama kelas kanonik: `non_cyberbullying`/`cyberbullying`, `non_buzzer`/`buzzer_like`.
- Seed wajib: 42, 123, 2024. Tidak ada randomness tanpa seed.
- Dataset mentah tidak di-commit; hanya dataset card dan skrip unduh.
- Split: stratified 80/10/10 dengan group split berdasarkan `group_id` bila tersedia.
- Metrik utama: macro-F1. Metrik efisiensi: parameter, ukuran model, waktu training, latency.
- Semua path berasal dari root proyek; konfigurasi tidak memuat path absolut.
- Tidak ada dependency opsional yang di-hardcode: `transformers`/`torch` diimpor lazy di modul yang membutuhkannya agar unit test data tetap ringan.

## Review Focus

- File CSV sumber dengan encoding campuran (UTF-8 BOM, Latin-1) dan delimiter berbeda harus terbaca tanpa crash.
- Label mentah yang tidak dikenal harus menghasilkan `None` dan dilaporkan, bukan diam-diam menjadi kelas 0.
- Near-duplicate tidak boleh bocor antara train/val/test.
- Korpus tanpa `user_id` tetap dapat diproses dengan group split berbasis klaster teks.
- Teks kosong setelah cleaning harus dibuang sebelum training.

---

### Task 1: Scaffold, config, IO, dan seeding

**Files:**
- Create: `requirements.txt`, `src/__init__.py`, `src/utils/__init__.py`, `src/utils/config.py`, `src/utils/io.py`, `src/utils/seed.py`
- Test: `tests/test_utils.py`

**Interfaces:**
- Produces:
  - `load_config(path: str | Path) -> dict`
  - `project_root() -> Path`
  - `resolve_path(value: str, root: Path | None = None) -> Path`
  - `read_table(path: str | Path) -> pd.DataFrame`
  - `write_json(obj: Any, path: str | Path) -> None`
  - `read_json(path: str | Path) -> Any`
  - `set_seed(seed: int) -> None`

- [ ] **Step 1: Write failing tests** untuk `load_config` membaca YAML, `read_table` membaca CSV/TSV/JSONL dan encoding fallback, serta `set_seed` deterministik.
- [ ] **Step 2: Run pytest, pastikan gagal karena modul belum ada.**
- [ ] **Step 3: Implementasikan** keempat modul. `read_table` mencoba encoding `utf-8-sig`, `utf-8`, lalu `latin-1`; deteksi delimiter dengan `csv.Sniffer` atau ekstensi.
- [ ] **Step 4: Run pytest, pastikan lulus.**
- [ ] **Step 5: Tulis `requirements.txt`** dengan versi minimum yang kompatibel Colab.

### Task 2: Cleaning dan harmonisasi label

**Files:**
- Create: `src/data/__init__.py`, `src/data/cleaning.py`, `src/data/labels.py`
- Test: `tests/test_cleaning.py`, `tests/test_labels.py`

**Interfaces:**
- Produces:
  - `clean_text(text: str, keep_emoji: bool = True) -> str`
  - `normalize_slang(text: str) -> str`
  - `harmonize_label(raw: object, task: str) -> int | None`
  - `TASK_LABELS: dict[str, dict[int, str]]`

- [ ] **Step 1: Write failing tests** untuk case folding, penghapusan URL/mention, normalisasi singkatan umum (`gk`->`tidak`, `yg`->`yang`), dan pemetaan label (`CB`, `Non_CB`, `-1`, `1`, `negative`, `positive`, label tak dikenal -> `None`).
- [ ] **Step 2: Run pytest, pastikan gagal.**
- [ ] **Step 3: Implementasikan** cleaning dengan regex dan kamus slang kecil; harmonisasi label dengan peta eksplisit per task.
- [ ] **Step 4: Run pytest, pastikan lulus.**

### Task 3: Sumber data dan persiapan dataset

**Files:**
- Create: `src/data/sources.py`, `src/data/prepare.py`, `configs/data/cyberbullying.yaml`, `configs/data/buzzer.yaml`
- Test: `tests/test_prepare.py`

**Interfaces:**
- Consumes: `read_table`, `clean_text`, `harmonize_label`.
- Produces:
  - `download_public_sources(dest_dir: str | Path) -> dict[str, Path]`
  - `load_source(spec: dict) -> pd.DataFrame` (kolom `text`, `label`, `source`, `group_id`)
  - `prepare_task(config: dict) -> dict[str, Path]` (menulis `train.csv`, `val.csv`, `test.csv`, `stats.json`)

- [ ] **Step 1: Write failing test** yang membuat CSV sintetis dengan encoding Latin-1 dan label campuran, lalu memastikan `prepare_task` menghasilkan tiga split, membuang teks kosong, dan mencatat label tak dikenal.
- [ ] **Step 2: Run pytest, pastikan gagal.**
- [ ] **Step 3: Implementasikan** `sources.py` dengan URL mentah GitHub untuk tiga sumber utama, plus mode `path` manual untuk Kaggle. `prepare.py` menggabungkan sumber, membersihkan, deduplikasi, dan memanggil splitter.
- [ ] **Step 4: Run pytest, pastikan lulus.**

### Task 4: Weak labeling buzzer

**Files:**
- Create: `src/data/weak_label.py`
- Test: `tests/test_weak_label.py`

**Interfaces:**
- Produces:
  - `near_duplicate_clusters(texts: Sequence[str], threshold: float = 0.85) -> list[int]`
  - `weak_label_frame(df: pd.DataFrame, config: dict) -> pd.DataFrame` dengan kolom `label`, `weak_score`, `cluster_id`, `group_id`

- [ ] **Step 1: Write failing tests** untuk dua teks identik mendapat klaster sama, teks berbeda mendapat klaster berbeda, dan `weak_label_frame` memberi label 1 pada teks duplikat bertemplate serta 0 pada teks unik.
- [ ] **Step 2: Run pytest, pastikan gagal.**
- [ ] **Step 3: Implementasikan** TF-IDF + cosine similarity, clustering union-find, skor gabungan (klaster, hashtag/mention/link, leksikon), ambang dari config.
- [ ] **Step 4: Run pytest, pastikan lulus.**

### Task 5: Split tanpa leakage

**Files:**
- Create: `src/data/split.py`
- Test: `tests/test_split.py`

**Interfaces:**
- Produces: `stratified_group_split(df: pd.DataFrame, label_col: str, group_col: str, ratios: dict[str, float], seed: int) -> dict[str, pd.DataFrame]`

- [ ] **Step 1: Write failing test** yang memastikan semua `group_id` hanya muncul di satu split dan proporsi label mendekati target.
- [ ] **Step 2: Run pytest, pastikan gagal.**
- [ ] **Step 3: Implementasikan** group-aware stratified split (kelompokkan per grup, stratifikasi berdasarkan label dominan grup).
- [ ] **Step 4: Run pytest, pastikan lulus.**

### Task 6: Loader model dan tokenisasi

**Files:**
- Create: `src/models/__init__.py`, `src/models/loader.py`, `src/data/tokenize.py`
- Test: `tests/test_tokenize.py` (tanpa torch, gunakan dummy tokenizer)

**Interfaces:**
- Produces:
  - `load_model_and_tokenizer(model_name: str, num_labels: int, cache_dir: str | None = None) -> tuple[Any, Any]`
  - `count_parameters(model: Any) -> int`
  - `model_disk_size_mb(model: Any) -> float`
  - `tokenize_splits(splits: dict[str, pd.DataFrame], tokenizer: Any, max_length: int, text_col: str = "text", label_col: str = "label") -> Any` (mengembalikan `DatasetDict`)

- [ ] **Step 1: Write failing test** dengan dummy tokenizer yang mengembalikan `input_ids`/`attention_mask`, memastikan jumlah baris dan label tetap.
- [ ] **Step 2: Run pytest, pastikan gagal.**
- [ ] **Step 3: Implementasikan** loader lazy-import dan tokenisasi.
- [ ] **Step 4: Run pytest, pastikan lulus.**

### Task 7: Training harness dan metrik

**Files:**
- Create: `src/training/__init__.py`, `src/training/metrics.py`, `src/training/trainer.py`
- Test: `tests/test_metrics.py`
- Config: `configs/cyberbullying/*.yaml`, `configs/buzzer/*.yaml`

**Interfaces:**
- Produces:
  - `compute_metrics(eval_pred: tuple[Any, Any]) -> dict[str, float]` (accuracy, macro/weighted P/R/F1, ROC-AUC bila biner)
  - `build_class_weights(labels: Sequence[int], num_labels: int) -> Any`
  - `run_experiment(config: dict) -> dict` (menulis `metrics.json`, `efficiency.json`, checkpoint)

- [ ] **Step 1: Write failing test** `compute_metrics` dengan logits sintetis dan label biner/multikelas.
- [ ] **Step 2: Run pytest, pastikan gagal.**
- [ ] **Step 3: Implementasikan** metrik dan trainer HF dengan early stopping, class weights, fp16 opsional, serta pengukuran efisiensi.
- [ ] **Step 4: Run pytest, pastikan lulus.**
- [ ] **Step 5: Buat 10 file config** (5 model x 2 tugas) dengan nilai dari spec.

### Task 8: Agregasi hasil dan plot

**Files:**
- Create: `src/evaluation/__init__.py`, `src/evaluation/collect.py`, `src/evaluation/plots.py`, `scripts/collect_results.py`
- Test: `tests/test_collect.py`

**Interfaces:**
- Produces:
  - `collect_metrics(results_dir: str | Path) -> pd.DataFrame`
  - `aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame`
  - `plot_comparisons(metrics: pd.DataFrame, figures_dir: str | Path) -> list[Path]`

- [ ] **Step 1: Write failing test** yang membuat dua file `metrics.json` sintetis dan memastikan agregasi menghasilkan mean/std per model/task.
- [ ] **Step 2: Run pytest, pastikan gagal.**
- [ ] **Step 3: Implementasikan** pembacaan artefak, agregasi, dan plot batang/box.
- [ ] **Step 4: Run pytest, pastikan lulus.**

### Task 9: CLI scripts

**Files:**
- Create: `scripts/prepare_data.py`, `scripts/run_experiment.py`, `scripts/predict.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: fungsi dari Task 3, 7, dan loader.
- Produces: CLI `--config`, `--task`, `--text`, `--model-dir`.

- [ ] **Step 1: Write failing test** yang menjalankan `python scripts/prepare_data.py --help` dan `--config` dengan file sementara.
- [ ] **Step 2: Run pytest, pastikan gagal.**
- [ ] **Step 3: Implementasikan** argparse dan pemanggilan fungsi inti.
- [ ] **Step 4: Run pytest, pastikan lulus.**

### Task 10: Demo Streamlit

**Files:**
- Create: `app/streamlit_app.py`, `app/__init__.py`
- Test: `tests/test_app_helpers.py`

**Interfaces:**
- Produces: `load_available_models(results_dir) -> list[dict]`, `predict_text(text, artifact_dir) -> dict`

- [ ] **Step 1: Write failing test** untuk `load_available_models` pada direktori kosong dan direktori dengan `metrics.json`.
- [ ] **Step 2: Run pytest, pastikan gagal.**
- [ ] **Step 3: Implementasikan** helper dan UI Streamlit (pilih tugas/model, input teks, probabilitas, tab perbandingan).
- [ ] **Step 4: Run pytest, pastikan lulus.**

### Task 11: Dokumentasi dan notebook

**Files:**
- Create: `README.md`, `data/DATASET_CARD.md`, `notebooks/01_eda.ipynb`, `notebooks/02_colab_train.ipynb`, `.gitignore`

- [ ] **Step 1: Tulis README** berisi setup, cara unduh data, perintah menjalankan, struktur hasil, dan batasan.
- [ ] **Step 2: Tulis dataset card** berisi sumber, lisensi, jumlah, skema label, dan limitation weak-label.
- [ ] **Step 3: Buat notebook Colab** yang meng-install dependency, mount Drive, menjalankan prepare + run + collect.
- [ ] **Step 4: Verifikasi perintah di README benar-benar ada.**

### Task 12: Verifikasi end-to-end

- [ ] **Step 1: Jalankan `pytest -q`.**
- [ ] **Step 2: Jalankan `python scripts/prepare_data.py --config configs/data/cyberbullying.yaml` pada data publik.**
- [ ] **Step 3: Jalankan satu smoke run dengan model kecil pada subset, atau verifikasi `--help` bila GPU tidak tersedia.**
- [ ] **Step 4: Jalankan `python scripts/collect_results.py` dan pastikan tabel terbentuk.**
- [ ] **Step 5: Catat hasil verifikasi di README bagian "Status Verifikasi".**
