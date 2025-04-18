import sounddevice as sd
import numpy as np
import librosa
import scipy.spatial
import sqlite3
from scipy import signal
import matplotlib.pyplot as plt

SAMPLE_RATE = 16000
DURATION = 4

def record_audio(filename: str):
    print(f"🎙️ Recording for {DURATION} seconds...")
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    np.save(filename + ".npy", audio)
    print(f"✅ Audio saved to {filename}.npy")

def load_audio_from_npy(filename):
    return np.load(filename + ".npy")

def extract_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

def compare_embeddings(emb1, emb2):
    return 1 - scipy.spatial.distance.cosine(emb1, emb2)

def init_db():
    conn = sqlite3.connect("voice_auth.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, embedding BLOB)")
    conn.commit()
    conn.close()

def save_embedding_to_db(username, embedding):
    conn = sqlite3.connect("voice_auth.db")
    cursor = conn.cursor()
    blob = embedding.astype(np.float32).tobytes()
    cursor.execute("REPLACE INTO users (username, embedding) VALUES (?, ?)", (username, blob))
    conn.commit()
    conn.close()

def load_embedding_from_db(username):
    conn = sqlite3.connect("voice_auth.db")
    cursor = conn.cursor()
    cursor.execute("SELECT embedding FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return np.frombuffer(result[0], dtype=np.float32)
    return None

def denoise_audio(audio):
    freqs, times, spec = signal.spectrogram(audio, SAMPLE_RATE)
    spec_denoised = np.where(spec < np.median(spec), 0, spec)
    _, audio_denoised = signal.istft(spec_denoised, SAMPLE_RATE)
    return audio_denoised

def is_live(audio):
    energy = np.sum(audio ** 2) / len(audio)
    silence_ratio = np.sum(np.abs(audio) < 0.02) / len(audio)
    return energy > 1e-4 and silence_ratio < 0.6

def plot_score(score):
    plt.bar(["Similarity Score"], [score], color="green" if score > 0.75 else "red")
    plt.ylim(0, 1)
    plt.title("Voice Match Confidence")
    plt.ylabel("Score")
    plt.show()

def log_history(username, score, success, message):
    conn = sqlite3.connect("voice_auth.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            score REAL,
            success INTEGER,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute(
        "INSERT INTO history (username, score, success, message) VALUES (?, ?, ?, ?)",
        (username, score, int(success), message)
    )
    conn.commit()
    conn.close()

def get_user_history(username):
    conn = sqlite3.connect("voice_auth.db")
    cursor = conn.cursor()
    cursor.execute("SELECT score, success, message, timestamp FROM history WHERE username = ? ORDER BY timestamp DESC", (username,))
    results = cursor.fetchall()
    conn.close()
    return [
        {"score": r[0], "success": bool(r[1]), "message": r[2], "timestamp": r[3]}
        for r in results
    ]