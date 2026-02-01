import torch
import hashlib
import pickle
import os
from onyx_arch import OnyxModel

# --- ALIGNED CONFIG ---
checkpoint_path = 'Checkpoints/onyx_model.pth'
vocab_path = 'Onyx_Vocab.pkl'
original_file = 'Pure_Logic/image_logic_01.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 256

# 1. SETUP & VOCAB LOADING
if not os.path.exists(vocab_path):
    print("❌ Error: Vocab file not found!")
    exit()

with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

# Create the inverse mapping (itos) from the vocab dict
itos = {i: ch for ch, i in vocab.items()}
stoi = vocab
vocab_size = len(vocab)

# 2. LOAD MODEL
model = OnyxModel(vocab_size).to(device)
if os.path.exists(checkpoint_path):
    # Loading the state_dict directly as saved by your training script
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    print(f"🔐 Expert Brain Loaded. Ready for Integrity Check.")
else:
    print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
    exit()

# 3. LOAD & PREP REFERENCE DATA
with open(original_file, 'r', encoding='latin-1') as f:
    clean_data = f.read()
print(f"ℹ️  Target Data Size: {len(clean_data)} bytes")

# 4. RECONSTRUCTION SEQUENCE
# Prime the context with the first 'block_size' real bytes
key_indices = [stoi[c] for c in clean_data[:block_size]]
context = torch.tensor([key_indices], dtype=torch.long, device=device)
generated_indices = list(key_indices)

print(f"⏳ Reconstructing from Neural Memory (Sequential Pass)...")

# 5. GENERATION LOOP (BIT-FOR-BIT)
with torch.no_grad():
    # We generate the rest of the file based on the initial 256-byte seed
    for i in range(len(clean_data) - block_size):
        logits = model(context)
        # Pull only the last logit for the next prediction
        next_idx = torch.argmax(logits[:, -1, :], dim=-1)
        generated_indices.append(next_idx.item())
        
        # Slide window: remove first, append the predicted next_idx
        context = torch.cat((context[:, 1:], next_idx.unsqueeze(0)), dim=1)
        
        if i % 1000 == 0:
            print(f"\rProgress: {i}/{len(clean_data)-block_size} bytes", end="")

print("\n✅ Reconstruction Complete.")

# 6. SHA-256 VERIFICATION
gen_str = "".join([itos[i] for i in generated_indices])

def get_sha256(s):
    return hashlib.sha256(s.encode('latin-1')).hexdigest()

real_hash = get_sha256(clean_data)
ai_hash = get_sha256(gen_str)

print(f"\n🛡️ SHA-256 REPORT:")
print(f"ORIGINAL: {real_hash}")
print(f"ONYX AI : {ai_hash}")

if real_hash == ai_hash:
    print("\n🚀 SINGULARITY CONFIRMED: 100% SHA-256 MATCH.")
    print("Zero collisions. Zero hallucinations. Perfect neural storage.")
else:
    print("\n❌ INTEGRITY FAILURE.")
    # Finder: Locate the exact point where the AI "hallucinated"
    for i, (a, b) in enumerate(zip(clean_data, gen_str)):
        if a != b:
            print(f"💀 Fatal Error at Byte {i}: Expected '{a}' -> Got '{b}'")
            break