from voice_utils import *

init_db()

user = input("👤 Enter username to register: ").strip()
if not user:
    print("Invalid username.")
    exit()

record_audio(user)
audio = load_audio_from_npy(user)

if not is_live(audio):
    print("Liveness check failed. Try again.")
    exit()

audio = denoise_audio(audio)
embedding = extract_features(audio)
save_embedding_to_db(user, embedding)
print(f"✅ Voice registered for user '{user}'.")
