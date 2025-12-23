#!/usr/bin/env python3
"""
Simple test script for Chatterbox TTS
"""

import torch
import torchaudio as ta

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

try:
    from chatterbox.vc import ChatterboxVC
    print("[OK] Chatterbox VC imported successfully!")

    # Try to load the model (this will download if needed)
    print("Loading Chatterbox VC model...")
    model = ChatterboxVC.from_pretrained(device)
    print("[OK] Chatterbox VC model loaded successfully!")

    print("\nSUCCESS: Chatterbox is ready to use!")
    print("You can now run:")
    print("  python example_vc.py")
    print("  python example_tts.py")
    print("  python gradio_vc_app.py")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
