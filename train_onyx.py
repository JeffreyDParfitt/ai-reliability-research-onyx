import torch
import torch.nn as nn
import pickle
import os
import sys
from onyx_arch import OnyxModel

# --- CONFIG ---
file_path = 'Pure_Logic/image_logic_01.txt'
vocab_path = 'Onyx_Vocab.pkl'
checkpoint_path = 'Checkpoints/onyx_model.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 256
learning_rate = 1e-7  

# 1. LOAD DATA & VOCAB
with open(file_path, 'r', encoding='latin-1') as f:
    text = f.read()

if not os.path.exists(vocab_path):
    print("❌ Error: Vocab file missing. Run initialization first.")
    sys.exit()

with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

vocab_size = len(vocab)
data_tensor = torch.tensor([vocab[c] for c in text], dtype=torch.long).to(device)

# 2. INITIALIZE MODEL
model = OnyxModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Load existing weights to continue from where you left off
if os.path.exists(checkpoint_path):
    print("🧠 Loading existing weights for refinement...")
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

# 3. THE SLIDING MICRO-DRILL LOOP
print(f"🚀 Starting Sliding Micro-Drill. Data Size: {len(data_tensor)} bytes.")
print("🎯 Targeting zero-error at Byte 257 transition.")

num_segments = len(data_tensor) - block_size
epoch = 0

try:
    while True:
        epoch += 1
        total_loss = 0
        
        # SLIDE BY 1 BYTE (The Handshake Fix)
        for i in range(0, num_segments, 1):
            model.train()
            
            # Input (0-255), Target (1-256) -> Next Byte prediction
            x = data_tensor[i:i+block_size].unsqueeze(0)
            y = data_tensor[i+1:i+block_size+1].unsqueeze(0)
            
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Monitor progress every 500 steps
            if i % 500 == 0:
                print(f"Epoch {epoch} | Byte {i}/{num_segments} | Loss: {loss.item():.10f}")
            
            # THE SINGULARITY TRIGGER
            if loss.item() < 1e-8:
                print(f"\n✨ SINGULARITY REACHED AT BYTE {i}!")
                torch.save(model.state_dict(), checkpoint_path)
                # If loss is ultra-low across a whole pass, we are done
        
        avg_loss = total_loss / num_segments
        print(f"💾 Sliding Pass {epoch} Complete. Avg Loss: {avg_loss:.10f}")
        torch.save(model.state_dict(), checkpoint_path)
        
        # Safety exit if we are bit-perfect
        if avg_loss < 1e-7:
            print("🏆 GLOBAL SINGULARITY REACHED. Terminal sequence locked.")
            break

except KeyboardInterrupt:
    print("\n🛑 Manual Stop. Checkpoint saved.")
    torch.save(model.state_dict(), checkpoint_path)