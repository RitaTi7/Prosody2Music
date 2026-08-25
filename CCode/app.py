"""
app.py — Interfaccia grafica per la pipeline Prosody2Music.

Avvio:
    streamlit run app.py

Gira in locale (è un'app Streamlit vera, non una pagina web statica),
quindi usa esattamente la stessa pipeline Python del progetto — PhonItalia,
Q2Stress, NRC EmoLex, corpus Lakh MIDI, il transformer melodico addestrato —
nessuna riscrittura, solo un'interfaccia sopra a run_pipeline().
"""

import os
import glob
import time
import streamlit as st

from main import run_pipeline
from instruments import INSTRUMENT_PRESETS

DEMO_POEM = """Nel mezzo del cammin di nostra vita
mi ritrovai per una selva oscura
ché la diritta via era smarrita"""

OUT_DIR = "streamlit_output"
os.makedirs(OUT_DIR, exist_ok=True)

st.set_page_config(page_title="Prosody2Music", page_icon="🎼", layout="wide")

# --- stile minimo, solo per rendere le card un po' più curate ---
st.markdown("""
<style>
.metric-box {
    background: #f6f5f3; border-radius: 10px; padding: 0.9rem 1rem;
    border: 1px solid #e5e3df; text-align: center;
}
.metric-box .label { font-size: 0.78rem; color: #6b6b6b; text-transform: uppercase; letter-spacing: 0.03em; }
.metric-box .value { font-size: 1.4rem; font-weight: 600; margin-top: 0.15rem; }
.stage-pill {
    display: inline-block; padding: 0.15rem 0.65rem; border-radius: 999px;
    background: #eef2ff; color: #3730a3; font-size: 0.78rem; margin-right: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🎼 Prosody2Music")
st.caption(
    "Poesia → analisi prosodica (PhonItalia + Q2Stress) → emotion embedding (NRC + lessico) "
    "→ Music Transformer (corpus Lakh MIDI) → orchestra a 4 strumenti → MIDI + audio"
)

# ============================== SIDEBAR: INPUT ==============================
with st.sidebar:
    st.header("Testo")

    example_files = sorted(glob.glob("*.txt"))
    example_choice = st.selectbox(
        "Poesia di esempio", ["(scrivi la tua)"] + example_files, index=0
    )

    if example_choice != "(scrivi la tua)":
        with open(example_choice, "r", encoding="utf-8") as f:
            default_text = f.read()
    else:
        default_text = DEMO_POEM

    poem_text = st.text_area("Testo della poesia (un verso per riga)", value=default_text, height=200)

    uploaded = st.file_uploader("...oppure carica un file .txt", type=["txt"])
    if uploaded is not None:
        poem_text = uploaded.read().decode("utf-8")

    st.divider()
    st.header("Parametri musicali")

    auto_tempo = st.checkbox("Tempo automatico (dall'arousal)", value=True)
    tempo = None
    if not auto_tempo:
        tempo = st.slider("Tempo (BPM)", 40, 180, 90)

    duration_scale = st.slider("Scala durata", 0.25, 4.0, 1.0, step=0.25)

    st.divider()
    st.header("Strumenti")

    instrument_labels = {k: v["label"] for k, v in INSTRUMENT_PRESETS.items()}
    inst_options = ["Auto (NLP)"] + sorted(instrument_labels, key=lambda k: instrument_labels[k])

    def _fmt_inst(key):
        return "Auto (NLP)" if key == "Auto (NLP)" else instrument_labels[key]

    melody_choice = st.selectbox("Voce principale", inst_options, format_func=_fmt_inst)
    harmony_choice = st.selectbox("Accompagnamento", inst_options, format_func=_fmt_inst)

    st.divider()
    use_random_seed = st.checkbox("Seed casuale", value=True)
    seed = None if use_random_seed else st.number_input("Seed", value=42, step=1)

    generate = st.button("🎵 Genera", type="primary", use_container_width=True)

# ============================== GENERAZIONE ==============================
if generate:
    if not poem_text.strip():
        st.error("Il testo della poesia è vuoto.")
        st.stop()

    melody_override = None if melody_choice == "Auto (NLP)" else melody_choice
    harmony_override = None if harmony_choice == "Auto (NLP)" else harmony_choice
    basename = f"run_{int(time.time())}"

    with st.spinner("Analisi prosodica, emotiva e generazione musicale in corso..."):
        try:
            midi_path, wav_path, meta, emotion = run_pipeline(
                poem_text,
                out_dir=OUT_DIR,
                basename=basename,
                seed=seed,
                verbose=False,
                tempo_override=tempo,
                duration_scale=duration_scale,
                melody_instrument_override=melody_override,
                harmony_instrument_override=harmony_override,
            )
        except ValueError as e:
            st.error(f"Parametro non valido: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Errore durante la generazione: {e}")
            st.stop()

    st.session_state["last_result"] = {
        "midi_path": midi_path,
        "wav_path": wav_path,
        "wav_fluid_path": os.path.join(OUT_DIR, f"{basename}_fluid.wav"),
        "meta": meta,
        "emotion": emotion,
        "melody_plot": os.path.join(OUT_DIR, "melody_rhythm.png"),
        "emotion_plot": os.path.join(OUT_DIR, "emotion_space.png"),
    }

# ============================== OUTPUT ==============================
result = st.session_state.get("last_result")

if result is None:
    st.info("Scrivi o scegli una poesia nel pannello a sinistra, poi premi **Genera**.")
else:
    emo = result["emotion"]
    meta = result["meta"]

    st.subheader("Analisi emotiva")
    c1, c2, c3 = st.columns(3)
    for col, label, value in zip(
        (c1, c2, c3),
        ("Valenza", "Arousal", "Tenerezza"),
        (emo["valence"], emo["arousal"], emo["tenderness"]),
    ):
        col.markdown(
            f'<div class="metric-box"><div class="label">{label}</div>'
            f'<div class="value">{value:+.2f}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<span class="stage-pill">Modalità: {meta["mode"]}</span>'
        f'<span class="stage-pill">Tempo: {meta["tempo"]} BPM</span>'
        f'<span class="stage-pill">Fonte melodia: {meta.get("melody_source", "n/d")}</span>',
        unsafe_allow_html=True,
    )

    st.divider()

    col_plot1, col_plot2 = st.columns(2)
    if os.path.isfile(result["melody_plot"]):
        col_plot1.image(result["melody_plot"], caption="Profilo melodico e ritmo prosodico", use_container_width=True)
    if os.path.isfile(result["emotion_plot"]):
        col_plot2.image(result["emotion_plot"], caption="Spazio emotivo (valenza/arousal)", use_container_width=True)

    st.divider()
    st.subheader("Audio")

    tab_synth, tab_fluid = st.tabs(["Synth interno", "FluidSynth (se disponibile)"])

    with tab_synth:
        if os.path.isfile(result["wav_path"]):
            with open(result["wav_path"], "rb") as f:
                wav_bytes = f.read()
            st.audio(wav_bytes, format="audio/wav")
            dl1, dl2 = st.columns(2)
            dl1.download_button("⬇️ Scarica WAV", wav_bytes, file_name=os.path.basename(result["wav_path"]))
            if os.path.isfile(result["midi_path"]):
                with open(result["midi_path"], "rb") as f:
                    midi_bytes = f.read()
                dl2.download_button("⬇️ Scarica MIDI", midi_bytes, file_name=os.path.basename(result["midi_path"]))
        else:
            st.warning("File audio non trovato.")

    with tab_fluid:
        if os.path.isfile(result["wav_fluid_path"]):
            with open(result["wav_fluid_path"], "rb") as f:
                fluid_bytes = f.read()
            st.audio(fluid_bytes, format="audio/wav")
            st.download_button("⬇️ Scarica WAV (FluidSynth)", fluid_bytes,
                                file_name=os.path.basename(result["wav_fluid_path"]))
        else:
            st.caption("FluidSynth non ha prodotto un file su questa macchina "
                       "(binario/soundfont non disponibili) — resta comunque il synth interno.")
