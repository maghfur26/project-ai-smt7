# Dataset Card

Proyek: Analisis Komparatif Fine-Tuning SLM untuk Deteksi Buzzer dan Cyberbullying
Tanggal: 2026-11-13

## 1. Ringkasan

| Tugas | Sumber utama | Baris mentah | Setelah cleaning | Positif | Negatif |
|---|---|---:|---:|---:|---:|
| Cyberbullying | Instagram + TikTok (publik) | 2.208 | 2.191 | 1.018 | 1.173 |
| Buzzer-like | Tweet politik DKI 2017 | 900 | 890 | 120 (weak) | 770 |

Angka positif/negatif cyberbullying dihitung setelah deduplikasi dan harmonisasi
label, sebelum split. Angka pada `data/processed/<task>/stats.json` adalah sumber
kebenaran final untuk setiap kali pipeline dijalankan.

## 2. Sumber Cyberbullying

### 2.1 `cahkanor/id-cyberbullying-instagram`
- URL: https://github.com/cahkanor/id-cyberbullying-instagram
- Media: komentar Instagram
- Jumlah: 300 baris (150 CB, 150 Non-CB)
- Kolom: `no`, `text`, `label`
- Label asli: `CB`, `Non_CB`
- Lisensi: tidak dinyatakan eksplisit di repositori. Digunakan untuk penelitian
  dengan atribusi.
- Encoding: Latin-1/Windows-1252, delimiter tab.

### 2.2 `rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia`
- URL: https://github.com/rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia
- File: `dataset_komentar_instagram_cyberbullying.csv`
- Media: komentar Instagram
- Jumlah: 400 baris
- Kolom: `Id`, `Sentiment`, `Instagram Comment Text`
- Label asli: `negative`, `positive`
- Pemetaan: `negative` -> cyberbullying (1), `positive` -> non-cyberbullying (0)
- Lisensi: tidak dinyatakan eksplisit. Digunakan untuk penelitian dengan atribusi.
- Catatan: label asli adalah sentimen; pemetaan ke cyberbullying mengikuti
  penggunaan dataset dalam literatur deteksi cyberbullying.

### 2.3 `rhiosutoyo/Indonesian-TikTok-Cyberbullying-Comments-Dataset`
- URL: https://github.com/rhiosutoyo/Indonesian-TikTok-Cyberbullying-Comments-Dataset
- Media: komentar TikTok
- Jumlah: 1.508 baris
- Kolom: `sentiment`, `comment`
- Label asli: `-1` (cyberbullying), `1` (non-cyberbullying)
- Lisensi: MIT
- Paper: Prameswari et al. (2023), ICRAIE.

### 2.4 Sumber opsional (manual, Kaggle)
- `cttrhnn/cyberbullying-bahasa-indonesia` — komentar Instagram, ±650 baris,
  lisensi tidak diketahui.
- `leonss0711/cleaned-indonesian-cyberbullying-dataset` — gabungan tiga sumber,
  lisensi MIT, ±1.726 baris.
- `SEACrowd/id_multilabel_hs` — 13.169 tweet, multi-label hate speech/abusive,
  CC BY-NC-SA 4.0. Dapat dipakai sebagai suplemen dengan memetakan label
  `HS`/`Abusive` ke kelas positif.

Untuk menambahkan sumber Kaggle: unduh manual, letakkan di
`data/raw/cyberbullying/`, lalu tambahkan entri berikut ke
`configs/data/cyberbullying.yaml`:

```yaml
  - name: kaggle_manual
    path: data/raw/cyberbullying/nama_file.csv
    text_col: text
    label_col: label
    label_map:
      bullying: 1
      non-bullying: 0
    domain: instagram
```

## 3. Sumber Buzzer-like

### 3.1 `rizalespe` — Tweet Pilkada DKI 2017
- URL: https://github.com/rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia
- File: `dataset_tweet_sentiment_pilkada_DKI_2017.csv`
- Jumlah: 900 tweet
- Kolom: `Id`, `Sentiment`, `Pasangan Calon`, `Text Tweet`
- Tidak ada `user_id` atau `timestamp` pada file ini.
- Lisensi: tidak dinyatakan eksplisit. Digunakan untuk penelitian dengan atribusi.

Karena tidak ada dataset teks buzzer publik siap pakai, label dibentuk dengan
weak supervision. Pendekatan ini mengikuti temuan literatur bahwa buzzer
dicirikan oleh repetisi konten, koordinasi, dan intensitas kampanye, tetapi
**tidak** memverifikasi identitas atau perilaku akun.

### 3.2 Aturan Weak Label

Skor gabungan (0–1) dari fitur berikut:

| Fitur | Bobot | Definisi |
|---|---:|---|
| Klaster near-duplicate | 0,50 | Ukuran klaster teks dengan cosine similarity TF-IDF karakter >= 0,85 dan ukuran >= 3 |
| Kepadatan hashtag | 0,15 | `min(1, 10 * jumlah_hashtag / jumlah_token)` |
| Tautan | 0,15 | 1 jika teks mengandung URL, selain itu 0 |
| Leksikon | 0,20 | `min(1, 5 * jumlah_kata_leksikon / jumlah_token)` |

Ambang label: skor >= **0,30** -> `buzzer_like` (1), selain itu `non_buzzer` (0).
Leksikon mencakup kata seperti `ayo`, `sebarkan`, `viralkan`, `dukung`, `pilih`,
`follow`, `like`, `share`, `promo`, `provokasi`, dan sejenisnya.

Fitur hashtag/tautan dihitung dari teks mentah (`text_raw`), sedangkan klaster
near-duplicate dihitung dari teks yang sudah dibersihkan (`text`).

### 3.3 Group Split

Setiap klaster near-duplicate menjadi satu `group_id`. Split 80/10/10 dilakukan
pada level grup, sehingga varian teks dari klaster yang sama tidak muncul di dua
split berbeda. Untuk cyberbullying, near-duplicate juga dikelompokkan dengan
ambang 0,90.

## 4. Skema Label Kanonik

| Tugas | ID 0 | ID 1 |
|---|---|---|
| `cyberbullying` | `non_cyberbullying` | `cyberbullying` |
| `buzzer` | `non_buzzer` | `buzzer_like` |

Label mentah yang tidak dikenali dibuang dan dihitung pada
`stats.json:dropped_unknown_labels`.

## 5. Prapemrosesan

1. Unicode NFKC dan HTML unescape.
2. Case folding (lowercase).
3. Hapus URL, mention, hashtag, dan awalan RT.
4. Opsi hapus emoji.
5. Normalisasi singkatan/slang Indonesia (kamus kecil bawaan).
6. Ringkas karakter berulang dan whitespace.
7. Deduplikasi teks persis; teks yang sama dengan label bertentangan dibuang
   (7 baris pada cyberbullying) dan dicatat di `stats.json`.
8. Klaster near-duplicate untuk group split.

## 6. Batasan

- **Validitas buzzer**: label adalah proksi teks, bukan ground truth. Model tidak
  dapat membedakan buzzer berbayar dari akun organik yang kebetulan memakai
  template serupa.
- **Domain**: data cyberbullying didominasi Instagram/TikTok, sedangkan korpus
  buzzer berasal dari Twitter. Generalisasi lintas platform perlu diuji.
- **Ukuran**: dataset kecil (ribuan baris); hasil dapat sensitif terhadap split.
- **Ketidakseimbangan**: kelas positif buzzer hanya ±13%; macro-F1 dan class
  weighting dipakai, tetapi variansi tetap tinggi.
- **Lisensi**: beberapa sumber tidak menyatakan lisensi eksplisit. gunakan hanya
  untuk penelitian dan atribusi; jangan redistribusi tanpa izin.
- **Privasi**: teks publik dapat memuat nama pengguna; pipeline menghapus mention,
  tetapi tidak menjamin anonimitas sempurna.

## 7. Etika

- Tidak ada scraping langsung ke Instagram.
- Dataset hanya untuk penelitian dan edukasi.
- Jangan menggunakan model untuk menuduh individu sebagai buzzer.
- Moderasi konten nyata memerlukan peninjauan manusia dan kebijakan platform.
