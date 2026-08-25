"""
gui.py — GUI semplice (Tkinter, solo libreria standard) per Prosody2Music.

Avvolge main.run_pipeline() senza toccare la logica della pipeline: la GUI
importa le funzioni già esistenti (run_pipeline, INSTRUMENT_PRESETS) e le
chiama in un thread separato, cosi' la finestra non si blocca durante la
generazione (che puo' richiedere qualche secondo per via del Music
Transformer + FluidSynth).

Uso:
    python3 gui.py

Richiede le stesse dipendenze di main.py (mido, numpy, pandas, torch,
matplotlib, scipy, PyTorch, ecc.) piu' Pillow (PIL) SOLO per mostrare le
anteprime dei due grafici PNG dentro la finestra: se Pillow non e'
installato la GUI funziona lo stesso, semplicemente non mostra le
anteprime (i file .png restano comunque su disco, apribili col tasto
"Apri cartella output").
"""

import os
import sys
import io
import threading
import queue
import traceback
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Moduli della pipeline esistente: nessuna modifica, solo import.
from main import run_pipeline, DEMO_POEM, TEMPO_MIN, TEMPO_MAX, DURATION_SCALE_MIN, DURATION_SCALE_MAX
from instruments import INSTRUMENT_PRESETS

AUTO_LABEL = "(automatico)"
INSTRUMENT_CHOICES = [AUTO_LABEL] + sorted(INSTRUMENT_PRESETS)


class QueueWriter(io.TextIOBase):
    """File-like object che spinge ogni write() in una queue thread-safe,
    cosi' il thread di generazione puo' continuare a fare print() come
    main.py fa gia' (verbose=True) e il thread della GUI legge i messaggi
    con after() senza toccare direttamente i widget da un altro thread."""

    def __init__(self, q):
        self.q = q

    def write(self, s):
        if s:
            self.q.put(("log", s))
        return len(s)

    def flush(self):
        pass


class ProsodyGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Prosody2Music — GUI")
        self.geometry("980x680")
        self.minsize(820, 560)

        self.msg_queue = queue.Queue()
        self.worker = None
        self.last_out_dir = None
        self.last_paths = {}  # midi / wav / wav_fluid -> path

        self._build_widgets()
        self.after(100, self._poll_queue)

    # ---------------------------------------------------------------- UI --
    def _build_widgets(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        # Layout a due colonne: sinistra = input/parametri, destra = log + anteprime
        left = ttk.Frame(root)
        left.pack(side="left", fill="both", expand=False, padx=(0, 10))
        right = ttk.Frame(root)
        right.pack(side="right", fill="both", expand=True)

        # --- Testo poesia -----------------------------------------------
        ttk.Label(left, text="Testo (poesia / frase):").pack(anchor="w")
        self.text_poem = tk.Text(left, width=44, height=10, wrap="word")
        self.text_poem.insert("1.0", DEMO_POEM)
        self.text_poem.pack(fill="both", expand=True, pady=(0, 4))

        btn_row = ttk.Frame(left)
        btn_row.pack(fill="x", pady=(0, 10))
        ttk.Button(btn_row, text="Carica da file...", command=self._load_file).pack(side="left")
        ttk.Button(btn_row, text="Ripristina demo", command=self._reset_demo).pack(side="left", padx=6)

        # --- Parametri numerici -------------------------------------------
        params = ttk.LabelFrame(left, text="Parametri", padding=8)
        params.pack(fill="x", pady=(0, 10))

        self.var_tempo = tk.StringVar(value="")  # vuoto = auto (deciso dal Music Transformer)
        self.var_duration = tk.DoubleVar(value=1.0)
        self.var_seed = tk.StringVar(value="")
        self.var_basename = tk.StringVar(value="output")
        self.var_outdir = tk.StringVar(value=os.path.join(os.getcwd(), "output_gui"))

        row = 0
        ttk.Label(params, text=f"Tempo BPM ({TEMPO_MIN}-{TEMPO_MAX}, vuoto=auto):").grid(row=row, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.var_tempo, width=10).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(params, text=f"Duration scale ({DURATION_SCALE_MIN}-{DURATION_SCALE_MAX}):").grid(row=row, column=0, sticky="w")
        ttk.Scale(params, from_=DURATION_SCALE_MIN, to=DURATION_SCALE_MAX, orient="horizontal",
                  variable=self.var_duration, length=140).grid(row=row, column=1, sticky="w")
        self.lbl_duration = ttk.Label(params, text="1.00")
        self.lbl_duration.grid(row=row, column=2, sticky="w")
        self.var_duration.trace_add("write", lambda *a: self.lbl_duration.config(
            text=f"{self.var_duration.get():.2f}"))
        row += 1

        ttk.Label(params, text="Seed (vuoto=casuale):").grid(row=row, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.var_seed, width=10).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(params, text="Nome base output:").grid(row=row, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.var_basename, width=18).grid(row=row, column=1, columnspan=2, sticky="w")
        row += 1

        ttk.Label(params, text="Cartella output:").grid(row=row, column=0, sticky="w")
        out_row = ttk.Frame(params)
        out_row.grid(row=row, column=1, columnspan=2, sticky="we")
        ttk.Entry(out_row, textvariable=self.var_outdir, width=22).pack(side="left", fill="x", expand=True)
        ttk.Button(out_row, text="...", width=3, command=self._choose_outdir).pack(side="left")

        # --- Strumenti ------------------------------------------------------
        instr = ttk.LabelFrame(left, text="Strumenti", padding=8)
        instr.pack(fill="x", pady=(0, 10))

        ttk.Label(instr, text="Voce principale (melodia):").grid(row=0, column=0, sticky="w")
        self.var_melody_instr = tk.StringVar(value=AUTO_LABEL)
        ttk.Combobox(instr, textvariable=self.var_melody_instr, values=INSTRUMENT_CHOICES,
                     state="readonly", width=20).grid(row=0, column=1, sticky="w")

        ttk.Label(instr, text="Accompagnamento (armonia):").grid(row=1, column=0, sticky="w")
        self.var_harmony_instr = tk.StringVar(value=AUTO_LABEL)
        ttk.Combobox(instr, textvariable=self.var_harmony_instr, values=INSTRUMENT_CHOICES,
                     state="readonly", width=20).grid(row=1, column=1, sticky="w")

        # --- Genera -----------------------------------------------------
        self.btn_generate = ttk.Button(left, text="Genera musica", command=self._on_generate)
        self.btn_generate.pack(fill="x", pady=(0, 4))
        self.progress = ttk.Progressbar(left, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 10))

        # --- Output (file generati) --------------------------------------
        out_frame = ttk.LabelFrame(left, text="Ultimo output", padding=8)
        out_frame.pack(fill="x")
        self.btn_open_dir = ttk.Button(out_frame, text="Apri cartella output", command=self._open_outdir, state="disabled")
        self.btn_open_dir.pack(fill="x", pady=2)
        self.btn_play_synth = ttk.Button(out_frame, text="Riproduci WAV (synth interno)",
                                          command=lambda: self._open_file("wav"), state="disabled")
        self.btn_play_synth.pack(fill="x", pady=2)
        self.btn_play_fluid = ttk.Button(out_frame, text="Riproduci WAV (FluidSynth)",
                                          command=lambda: self._open_file("wav_fluid"), state="disabled")
        self.btn_play_fluid.pack(fill="x", pady=2)

        # --- Colonna destra: log + anteprime immagini ----------------------
        ttk.Label(right, text="Log:").pack(anchor="w")
        log_frame = ttk.Frame(right)
        log_frame.pack(fill="both", expand=True)
        self.text_log = tk.Text(log_frame, height=16, state="disabled", wrap="word", bg="#111", fg="#ddd")
        log_scroll = ttk.Scrollbar(log_frame, command=self.text_log.yview)
        self.text_log.configure(yscrollcommand=log_scroll.set)
        self.text_log.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        ttk.Label(right, text="Anteprime (ritmo/melodia e spazio emotivo):").pack(anchor="w", pady=(10, 0))
        self.preview_frame = ttk.Frame(right)
        self.preview_frame.pack(fill="both", expand=True)
        self.lbl_preview1 = ttk.Label(self.preview_frame, text="(nessuna generazione ancora)")
        self.lbl_preview1.pack(side="left", padx=5, pady=5)
        self.lbl_preview2 = ttk.Label(self.preview_frame, text="")
        self.lbl_preview2.pack(side="left", padx=5, pady=5)
        self._preview_imgs = []  # riferimenti forti, altrimenti Tkinter le fa sparire (garbage collected)

    # ------------------------------------------------------------- azioni --
    def _load_file(self):
        path = filedialog.askopenfilename(title="Scegli un file di testo",
                                           filetypes=[("Testo", "*.txt"), ("Tutti i file", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            messagebox.showerror("Errore lettura file", str(e))
            return
        self.text_poem.delete("1.0", "end")
        self.text_poem.insert("1.0", content)

    def _reset_demo(self):
        self.text_poem.delete("1.0", "end")
        self.text_poem.insert("1.0", DEMO_POEM)

    def _choose_outdir(self):
        path = filedialog.askdirectory(title="Scegli cartella di output")
        if path:
            self.var_outdir.set(path)

    def _log(self, msg):
        self.text_log.configure(state="normal")
        self.text_log.insert("end", msg)
        self.text_log.see("end")
        self.text_log.configure(state="disabled")

    def _on_generate(self):
        if self.worker and self.worker.is_alive():
            return  # generazione gia' in corso

        poem_text = self.text_poem.get("1.0", "end").strip()
        if not poem_text:
            messagebox.showwarning("Testo mancante", "Inserisci un testo o ripristina la poesia demo.")
            return

        # Validazione leggera lato GUI; run_pipeline rivalida comunque
        # (clamp automatico) quindi qui basta intercettare seed non numerico.
        seed_str = self.var_seed.get().strip()
        seed = None
        if seed_str:
            try:
                seed = int(seed_str)
            except ValueError:
                messagebox.showerror("Seed non valido", "Il seed deve essere un numero intero.")
                return

        tempo_str = self.var_tempo.get().strip()
        tempo_override = None
        if tempo_str:
            try:
                tempo_override = float(tempo_str)
            except ValueError:
                messagebox.showerror("Tempo non valido", "Il tempo deve essere un numero (BPM).")
                return

        melody_instr = self.var_melody_instr.get()
        harmony_instr = self.var_harmony_instr.get()
        melody_override = None if melody_instr == AUTO_LABEL else melody_instr
        harmony_override = None if harmony_instr == AUTO_LABEL else harmony_instr

        out_dir = self.var_outdir.get().strip() or "."
        basename = self.var_basename.get().strip() or "output"

        # Reset UI per la nuova run
        self.text_log.configure(state="normal")
        self.text_log.delete("1.0", "end")
        self.text_log.configure(state="disabled")
        self.btn_generate.configure(state="disabled", text="Generazione in corso...")
        self.btn_open_dir.configure(state="disabled")
        self.btn_play_synth.configure(state="disabled")
        self.btn_play_fluid.configure(state="disabled")
        self.progress.start(12)

        self.worker = threading.Thread(
            target=self._run_pipeline_thread,
            args=(poem_text, out_dir, basename, seed, tempo_override,
                  self.var_duration.get(), melody_override, harmony_override),
            daemon=True,
        )
        self.worker.start()

    def _run_pipeline_thread(self, poem_text, out_dir, basename, seed, tempo_override,
                              duration_scale, melody_override, harmony_override):
        writer = QueueWriter(self.msg_queue)
        old_stdout = sys.stdout
        sys.stdout = writer
        try:
            midi_path, wav_path, meta, emotion = run_pipeline(
                poem_text,
                out_dir=out_dir,
                basename=basename,
                seed=seed,
                verbose=True,
                tempo_override=tempo_override,
                duration_scale=duration_scale,
                melody_instrument_override=melody_override,
                harmony_instrument_override=harmony_override,
            )
            wav_fluid_path = os.path.join(out_dir, f"{basename}_fluid.wav")
            self.msg_queue.put(("done", {
                "out_dir": out_dir,
                "midi": midi_path,
                "wav": wav_path,
                "wav_fluid": wav_fluid_path,
            }))
        except Exception as e:
            self.msg_queue.put(("log", f"\n[ERRORE] {e}\n"))
            self.msg_queue.put(("log", traceback.format_exc()))
            self.msg_queue.put(("error", str(e)))
        finally:
            sys.stdout = old_stdout

    # ---------------------------------------------------------- polling --
    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "done":
                    self._on_generation_done(payload)
                elif kind == "error":
                    self._on_generation_error(payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _on_generation_done(self, paths):
        self.progress.stop()
        self.btn_generate.configure(state="normal", text="Genera musica")
        self.last_out_dir = paths["out_dir"]
        self.last_paths = paths
        self.btn_open_dir.configure(state="normal")
        if os.path.exists(paths["wav"]):
            self.btn_play_synth.configure(state="normal")
        if os.path.exists(paths["wav_fluid"]):
            self.btn_play_fluid.configure(state="normal")
        self._log("\n=== Generazione completata ===\n")
        self._show_previews(paths["out_dir"])

    def _on_generation_error(self, err):
        self.progress.stop()
        self.btn_generate.configure(state="normal", text="Genera musica")
        messagebox.showerror("Errore durante la generazione", err)

    def _show_previews(self, out_dir):
        if not HAS_PIL:
            self.lbl_preview1.configure(text="(Installa Pillow per vedere le anteprime: pip install Pillow)")
            return
        paths = [
            os.path.join(out_dir, "melody_rhythm.png"),
            os.path.join(out_dir, "emotion_space.png"),
        ]
        labels = [self.lbl_preview1, self.lbl_preview2]
        self._preview_imgs.clear()
        for path, label in zip(paths, labels):
            if not os.path.exists(path):
                label.configure(text=f"(non trovato: {os.path.basename(path)})", image="")
                continue
            try:
                img = Image.open(path)
                img.thumbnail((420, 420))
                photo = ImageTk.PhotoImage(img)
                self._preview_imgs.append(photo)  # tieni un riferimento vivo
                label.configure(image=photo, text="")
            except Exception as e:
                label.configure(text=f"(errore anteprima: {e})", image="")

    # ------------------------------------------------------------- utils --
    def _open_outdir(self):
        if self.last_out_dir:
            self._open_path(self.last_out_dir)

    def _open_file(self, key):
        path = self.last_paths.get(key)
        if path and os.path.exists(path):
            self._open_path(path)

    @staticmethod
    def _open_path(path):
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", path], check=False)
            else:
                subprocess.run(["xdg-open", path], check=False)
        except Exception as e:
            messagebox.showerror("Impossibile aprire", str(e))


if __name__ == "__main__":
    # Serve lanciare la GUI dalla cartella del progetto (dove stanno
    # main.py, instruments.py, ecc.) cosi' gli import sopra funzionano.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    app = ProsodyGUI()
    app.mainloop()
