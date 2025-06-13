import os
import csv
import datetime
import numpy as np
np.complex = complex  # For compatibility with some older librosa usage
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
from scipy.fft import rfft, rfftfreq

# --- Setup Directories ---
INPUT_DIR = "../input_clips"
OUTPUT_ROOT = "../outputs"
CSV_PATH = os.path.join(OUTPUT_ROOT, "../outputs/master_log.csv")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# --- Prepare CSV Master Log ---
write_header = not os.path.exists(CSV_PATH)
with open(CSV_PATH, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow([
            "filename",
            "duration_sec",
            "sample_rate",
            "inharmonicity_stddev",
            "peak_mod_freq_hz",
            "top_5_mod_freqs",
            "top_5_magnitudes",
            "timestamp"
        ])

# --- Get Audio Files ---
audio_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.wav', '.mp3', '.flac'))]
if not audio_files:
    print("No audio files found in ./inputs.")
    exit()

# --- Process Each File ---
for audio_file in audio_files:
    AUDIO_PATH = os.path.join(INPUT_DIR, audio_file)
    file_stem = os.path.splitext(audio_file)[0]
    OUTPUT_DIR = os.path.join(OUTPUT_ROOT, file_stem)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get file timestamp
    ts = os.path.getmtime(AUDIO_PATH)
    timestamp = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    # Load audio
    y, sr = librosa.load(AUDIO_PATH, sr=None)
    t = np.linspace(0, len(y) / sr, len(y))
    duration = len(y) / sr
    print(f"\nLoaded: {audio_file} | Duration: {duration:.2f}s | Sample rate: {sr} Hz")

    # Phase Inversion
    y_inv = -y
    interference = y + y_inv
    plt.figure(figsize=(12, 3))
    plt.plot(t, y, label="Original", alpha=0.6)
    plt.plot(t, y_inv, label="Inverted", alpha=0.6)
    plt.plot(t, interference, label="Sum (should cancel)", color='red')
    plt.title("Phase Inversion / Interference Mapping")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/phase_inversion.png", dpi=300)
    plt.close()

    # Cepstrum
    spectrum = np.fft.fft(y)
    log_mag = np.log1p(np.abs(spectrum))
    cepstrum = np.fft.ifft(log_mag).real
    plt.figure(figsize=(10, 3))
    plt.plot(cepstrum[:500])
    plt.title("Cepstral Analysis")
    plt.xlabel("Quefrency (samples)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cepstrum.png", dpi=300)
    plt.close()

    # Inharmonicity
    peaks, _ = librosa.piptrack(y=y, sr=sr)
    mean_peaks = peaks.mean(axis=1)
    nonzero_peaks = mean_peaks[mean_peaks > 0]
    plt.figure(figsize=(10, 3))
    plt.plot(nonzero_peaks)
    plt.title("Spectral Peaks (for Dissonance/Inharmonicity Estimation)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/spectral_peaks.png", dpi=300)
    plt.close()
    inharmonicity = np.std(nonzero_peaks)
    print(f"Inharmonicity proxy: std dev of spectral peaks = {inharmonicity:.2f} Hz")

    # Chromagram
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', cmap='coolwarm', sr=sr)
    plt.title("Chromagram")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/chromagram.png", dpi=300)
    plt.close()

      # Spectrogram (New Section)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram (Log-Frequency Scale)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/spectrogram.png", dpi=300)
    plt.close()

    # AM Envelope
    analytic_signal = hilbert(y)
    envelope = np.abs(analytic_signal)
    smoothed_env = gaussian_filter1d(envelope, sigma=500)
    plt.figure(figsize=(12, 4))
    plt.plot(y, alpha=0.4, label="Original Signal")
    plt.plot(smoothed_env, color="red", label="AM Envelope (Smoothed)")
    plt.title("Amplitude Modulation Envelope")
    plt.legend()
    plt.xlabel("Samples")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/am_envelope.png", dpi=300)
    plt.close()

    # Modulation Spectrum
    mod_spec = np.abs(rfft(smoothed_env))
    mod_freqs = rfftfreq(len(smoothed_env), 1 / sr)
    plt.figure(figsize=(10, 4))
    plt.plot(mod_freqs, mod_spec)
    plt.xlim([0, 50])
    plt.xlabel("Modulation Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Modulation Spectrum of AM Envelope")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/modulation_spectrum.png", dpi=300)
    plt.close()

    # Save raw data arrays
    np.save(f"{OUTPUT_DIR}/envelope.npy", smoothed_env)
    np.save(f"{OUTPUT_DIR}/mod_freqs.npy", mod_freqs)
    np.save(f"{OUTPUT_DIR}/mod_spec.npy", mod_spec)

    # Stats
    peak_mod_freq = mod_freqs[np.argmax(mod_spec)]
    print(f"Peak modulation frequency: {peak_mod_freq:.2f} Hz")

    top_idx = np.argsort(mod_spec[mod_freqs < 50])[::-1][:5]
    top_freqs = [mod_freqs[i] for i in top_idx]
    top_mags = [mod_spec[i] for i in top_idx]
    print("Top 5 modulation frequencies under 50 Hz:")
    for f, m in zip(top_freqs, top_mags):
        print(f"  {f:.2f} Hz — Magnitude: {m:.2f}")

    # Write row to master_log.csv
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            audio_file,
            round(duration, 2),
            sr,
            round(inharmonicity, 2),
            round(peak_mod_freq, 2),
            "|".join(f"{f:.2f}" for f in top_freqs),
            "|".join(f"{m:.2f}" for m in top_mags),
            timestamp
        ])
