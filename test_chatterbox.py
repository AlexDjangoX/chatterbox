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
    # Try the faster Turbo model first
    print("Testing Chatterbox Turbo TTS...")
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    print("Loading Turbo model...")
    model = ChatterboxTurboTTS.from_pretrained(device)
    print("[OK] Turbo model loaded successfully!")

    # Test with simple English text
    test_text = "Hello, this is a test of Chatterbox Turbo."
    print(f"Generating audio for: '{test_text}'")

    wav = model.generate(test_text)
    print(f"[OK] Audio generated! Shape: {wav.shape}, Sample rate: {model.sr}")

    # Save the test audio
    import torchaudio as ta
    ta.save("test_output.wav", wav, model.sr)
    print("[OK] Saved test audio to test_output.wav")

    print("\nSUCCESS: Chatterbox Turbo is working!")
    print("You can now run the web apps:")
    print("  python gradio_tts_turbo_app.py")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    import traceback
    traceback.print_exc()
