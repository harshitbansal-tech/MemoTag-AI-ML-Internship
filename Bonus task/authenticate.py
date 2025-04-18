from voice_utils import *

init_db()

user = input("👤 Enter username to authenticate: ").strip()
if not user:
    print("Invalid username.")
    exit()

ref_embed = load_embedding_from_db(user)
if ref_embed is None:
    print("User not found in database.")
    exit()

temp_file = "temp_test"
record_audio(temp_file)
audio = load_audio_from_npy(temp_file)

if not is_live(audio):
    print("Liveness check failed.")
    exit()

audio = denoise_audio(audio)
test_embed = extract_features(audio)

score = compare_embeddings(ref_embed, test_embed)
print(f"Similarity Score: {score:.4f}")
plot_score(score)

if score >= 0.75:
    print("Authentication Successful.")
else:
    print("Authentication Failed.")
