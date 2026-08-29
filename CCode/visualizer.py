"""
visualizer.py — Modulo per la generazione di grafici esplicativi per il progetto
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_training_loss(losses, save_path="output/plots/training_loss.png"):
    """Genera e salva il grafico della curva di Loss del Transformer."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(7, 4))
    
    epochs = list(range(1, len(losses) + 1))
    plt.plot(epochs, losses, marker='o', linewidth=2, color='#1f77b4', label='Training Loss')
    
    plt.title("Curva di Addestramento del Transformer (Fase 1)", fontsize=12, fontweight='bold')
    plt.xlabel("Epoca", fontsize=10)
    plt.ylabel("Cross-Entropy Loss", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(epochs)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[visualizer] Grafico loss salvato in: {save_path}")


def plot_melody_and_rhythm(melody, poem_analysis, save_path="output/plots/melody_rhythm.png"):
    """Genera un doppio grafico: profilo melodico MIDI in alto e accenti prosodici in basso."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    pitches = [n.pitch for n in melody.notes]
    durations = [n.duration for n in melody.notes]
    time_points = [0] + list(np.cumsum(durations)[:-1])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # 1. Grafico Pitch MIDI
    ax1.step(time_points, pitches, where='post', color='#2ca02c', linewidth=2, label="Pitch MIDI")
    ax1.scatter(time_points, pitches, color='#2ca02c', s=30, zorder=3)
    ax1.set_ylabel("Nota MIDI (Pitch)", fontsize=10)
    ax1.set_title("Profilo Melodico e Struttura Prosodica Generata", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc="upper right")

    # 2. Grafico Accenti Prosodici (Ritmo)
    stresses = []
    for verse in poem_analysis:
        for s in verse["syllables"]:
            stresses.append(2 if s["stressed"] else 1)

    time_stresses = time_points[:len(stresses)]
    ax2.bar(time_stresses, stresses, width=0.3, color='#ff7f0e', alpha=0.8, align='edge', label="Accento Prosodico")
    ax2.set_yticks([1, 2])
    ax2.set_yticklabels(['Atona (1)', 'Tonica (2)'])
    ax2.set_xlabel("Tempo Musicale (Beat)", fontsize=10)
    ax2.set_ylabel("Intensità", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[visualizer] Grafico melodia/ritmo salvato in: {save_path}")

#visualizer.py (gli import si trovano sopra)

def plot_emotion_space(emotion, mode, save_path="output/plots/emotion_space.png"):
    """Mappa i valori di Valenza e Arousal sul piano emotivo 2D."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6, 6))

    v = emotion["valence"]
    a = emotion["arousal"]

    # Assi principali
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)

    # Punto della poesia
    plt.scatter(v, a, color='#d62728', s=180, zorder=5, label=f"Poesia (Scala: {mode})")
    plt.annotate(f"({v:+.2f}, {a:+.2f})", (v + 0.04, a + 0.04), fontsize=10, fontweight='bold')

    # Etichette dei quadranti
    plt.text(0.5, 0.8, "Gioia / Eccitazione", fontsize=9, alpha=0.5, ha='center')
    plt.text(-0.5, 0.8, "Rabbia / Tensione", fontsize=9, alpha=0.5, ha='center')
    plt.text(-0.5, -0.8, "Tristezza / Malinconia", fontsize=9, alpha=0.5, ha='center')
    plt.text(0.5, -0.8, "Calma / Serenità", fontsize=9, alpha=0.5, ha='center')

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.xlabel("Valenza (Negativa ← → Positiva)", fontsize=10)
    plt.ylabel("Arousal (Calma ← → Intenso)", fontsize=10)
    plt.title("Spazio Emotivo V-A e Modalità Musicale", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[visualizer] Grafico spazio emotivo salvato in: {save_path}")