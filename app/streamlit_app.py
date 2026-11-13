"""Streamlit demo for cyberbullying and buzzer-like content detection."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.models.inference import load_available_models, predict_text  # noqa: E402
from src.utils.config import resolve_path  # noqa: E402

EXAMPLES = {
    "cyberbullying": [
        "Dasar bodoh, muka kayak apa pun pantas dihina.",
        "Keren banget fotonya, selamat ya atas prestasinya!",
        "Otak lo di mana? Gak usah sok pintar deh.",
    ],
    "buzzer": [
        "Ayo viralkan! Sebarkan ke semua grup, dukung terus sampai trending!",
        "Saya rasa diskusi ini perlu data yang lebih lengkap.",
        "Follow, like, dan share sebanyak mungkin. Jangan sampai kalah!",
    ],
}

TASK_TITLES = {
    "cyberbullying": "Deteksi Cyberbullying",
    "buzzer": "Deteksi Konten Bergaya Buzzer",
}


def _comparison_tab(results_dir: Path) -> None:
    metrics_path = results_dir / "metrics.csv"
    aggregate_path = results_dir / "tables" / "aggregate.csv"
    if not metrics_path.exists() and not aggregate_path.exists():
        st.info(
            "Belum ada hasil eksperimen. Jalankan training lalu "
            "`python scripts/collect_results.py`."
        )
        return

    if aggregate_path.exists():
        st.subheader("Rata-rata metrik per model")
        st.dataframe(pd.read_csv(aggregate_path), use_container_width=True)
    if metrics_path.exists():
        st.subheader("Semua run")
        st.dataframe(pd.read_csv(metrics_path), use_container_width=True)

    figures_dir = results_dir / "figures"
    for image in sorted(figures_dir.glob("*.png")):
        st.image(str(image), caption=image.stem.replace("_", " "))


def main() -> None:
    st.set_page_config(page_title="SLM Buzzer & Cyberbullying", layout="wide")
    st.title("Analisis Komparatif SLM")
    st.caption(
        "Demo deteksi cyberbullying dan konten bergaya buzzer berbahasa Indonesia. "
        "Label buzzer berasal dari weak supervision dan bersifat proksi."
    )

    results_dir = resolve_path("results")
    runs = load_available_models(results_dir / "models")
    if not runs:
        st.warning(
            "Belum ada model terlatih di `results/models/`. "
            "Jalankan `python scripts/prepare_data.py` lalu `python scripts/run_experiment.py`."
        )
        st.stop()

    tasks = sorted({run["task"] for run in runs if run.get("task")})
    task = st.sidebar.selectbox(
        "Tugas",
        tasks,
        format_func=lambda value: TASK_TITLES.get(value, value),
    )
    task_runs = [run for run in runs if run.get("task") == task]
    options = {
        f"{run['run_name']} (macro-F1 {run.get('macro_f1') or 0:.3f})": run
        for run in task_runs
    }
    selected_label = st.sidebar.selectbox("Model", list(options))
    selected = options[selected_label]

    st.sidebar.markdown(
        f"**Model:** `{selected['model']}`  \n"
        f"**Seed:** {selected['seed']}  \n"
        f"**Macro-F1:** {selected.get('macro_f1')}"
    )

    tab_predict, tab_compare = st.tabs(["Prediksi", "Perbandingan"])

    with tab_predict:
        example = st.selectbox("Contoh teks", EXAMPLES.get(task, []))
        text = st.text_area("Teks komentar", value=example, height=120)
        if st.button("Analisis", type="primary"):
            if not text.strip():
                st.error("Teks tidak boleh kosong.")
            else:
                with st.spinner("Menjalankan inferensi..."):
                    try:
                        result = predict_text(text, selected["model_dir"], task=task)
                    except Exception as error:  # pragma: no cover - UI guard
                        st.error(f"Gagal memuat model: {error}")
                        st.stop()
                st.subheader(f"Prediksi: {result['label']}")
                st.metric("Confidence", f"{result['confidence']:.3f}")
                for label, probability in result["probabilities"].items():
                    st.write(f"**{label}**: {probability:.3f}")
                    st.progress(min(1.0, max(0.0, probability)))

    with tab_compare:
        _comparison_tab(results_dir)


if __name__ == "__main__":
    main()
