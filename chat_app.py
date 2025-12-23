#!/usr/bin/env python3
"""
Real-time Chat App with Chatterbox TTS
"""

import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts_turbo import ChatterboxTurboTTS
import tempfile
import os

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model variable
model = None

def load_model():
    """Load the TTS model"""
    global model
    if model is None:
        print(f"Loading Chatterbox Turbo on {DEVICE}...")
        model = ChatterboxTurboTTS.from_pretrained(DEVICE)
    return model

def generate_response(message, history):
    """Generate response for chat message with optional TTS"""
    # Simple bot responses for demo
    responses = [
        f"I heard you say: '{message}'. That's interesting!",
        f"Thanks for your message: '{message}'. How can I help you today?",
        f"You mentioned: '{message}'. Tell me more about that!",
        f"Got your message: '{message}'. I'm here to chat!",
        f"That's a great point about '{message}'!",
        f"Hi there! You said '{message}'. What's on your mind?",
        f"'{message}' - I love hearing from you! What's next?",
    ]

    bot_response = random.choice(responses)

    # Try TTS if model is available
    audio_file = None
    try:
        if model is None:
            load_model()

        if model is not None:
            # Generate audio for the bot response
            wav = model.generate(bot_response)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_filename = tmp_file.name

            # Save the audio
            import torchaudio as ta
            ta.save(temp_filename, wav, model.sr)
            audio_file = temp_filename

    except Exception as e:
        # TTS failed, but we still have text response
        print(f"TTS failed: {e}")

    return bot_response, audio_file

def create_chat_interface():
    """Create the Gradio chat interface"""

    with gr.Blocks(title="Chatterbox Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 💬 Chatterbox Chat with Voice")
        gr.Markdown("Chat with AI that responds with both text and voice!")

        # Chat history
        chatbot = gr.Chatbot(height=400, show_label=False, type="messages")

        # Message input
        msg = gr.Textbox(
            placeholder="Type your message here...",
            show_label=False,
            container=False
        )

        # Audio output for TTS
        audio_output = gr.Audio(
            label="Bot Voice Response",
            visible=False  # Initially hidden
        )

        # Status
        status = gr.Textbox(
            label="Status",
            value="Loading TTS model...",
            interactive=False
        )

        # TTS status indicator
        tts_status = gr.Textbox(
            label="Voice Status",
            value="Checking...",
            interactive=False
        )

        def respond(message, history):
            if not message.strip():
                return history, None, "Ready to chat!", "Voice: Checking..."

            # Add user message to history
            history = history + [[message, None]]

            try:
                # Generate bot response
                bot_text, audio_file = generate_response(message, history)

                # Update history with bot response
                history[-1][1] = bot_text

                # Check TTS status
                if audio_file:
                    tts_msg = f"Voice: Active ({model.sr}Hz)"
                    status_msg = "Response generated with voice!"
                else:
                    tts_msg = "Voice: Offline (needs HF token)"
                    status_msg = "Response generated (text only)"

                return history, audio_file, status_msg, tts_msg

            except Exception as e:
                history[-1][1] = f"Sorry, I encountered an error: {str(e)}"
                return history, None, f"Error: {str(e)}", "Voice: Error"

        # Submit button
        msg.submit(
            respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, audio_output, status, tts_status],
            queue=True
        )

        # Clear button
        def clear_history():
            return [], None, "Chat cleared!", "Voice: Ready"

        gr.Button("Clear Chat").click(
            clear_history,
            outputs=[chatbot, audio_output, status, tts_status]
        )

        # Instructions
        gr.Markdown("""
        ### How to Use:
        1. Type your message and press Enter
        2. The AI will respond with text and voice
        3. Listen to the voice response in the audio player
        4. Continue chatting!

        ### Features:
        - Real-time text responses
        - Voice synthesis with Chatterbox Turbo
        - Conversation history
        - Natural-sounding AI voice

        ### Note:
        First response may take longer as the model loads.
        """)

    return demo

if __name__ == "__main__":
    demo = create_chat_interface()
    demo.queue(max_size=10, default_concurrency_limit=2)
    demo.launch(
        share=True,
        show_error=True,
        server_name="0.0.0.0"
        # Let Gradio find an available port automatically (removed server_port to auto-detect)
    )
