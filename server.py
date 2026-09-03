# ==============================================================================
#                      Pocket TTS Voice Synthesis Server
# ==============================================================================
# Standalone Flask server that exposes a '/tts' endpoint. It receives text and
# uses the Pocket TTS model to generate speech from a cloned reference voice.
# The generated audio is saved to tts_output/server_output.wav so that
# watcher_to_face.py can pick it up for facial animation (same as StyleTTS2).
# ==============================================================================

import os
import sys
import io
import configparser

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import numpy as np
import scipy.io.wavfile

from pocket_tts import TTSModel

# --- Load Configuration from settings.ini ---
config = configparser.ConfigParser()
config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.ini')
if not os.path.exists(config_file):
    sys.exit(f"FATAL ERROR: Configuration file '{config_file}' not found.")
config.read(config_file)

try:
    REFERENCE_VOICE = config.get('TTS', 'reference_voice')
    DEVICE = config.get('TTS', 'device', fallback='cpu')
    SERVER_HOST = config.get('Server', 'host')
    SERVER_PORT = config.getint('Server', 'port')
    SERVER_DEBUG = config.getboolean('Server', 'debug')
except (configparser.NoSectionError, configparser.NoOptionError) as e:
    sys.exit(f"FATAL ERROR: A required setting is missing in '{config_file}'. Details: {e}")

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Global model + voice state (loaded once) ---
tts_model = None
voice_state = None


def load_model():
    global tts_model
    if tts_model is None:
        import torch
        if DEVICE == "cuda" and not torch.cuda.is_available():
            print("⚠️ CUDA requested but not available (CPU-only PyTorch). Falling back to CPU.")
            effective_device = "cpu"
        else:
            effective_device = DEVICE
        print(f"Loading Pocket TTS model on device '{effective_device}'...")
        tts_model = TTSModel.load_model()
        if effective_device == "cuda":
            tts_model.to("cuda")
            print("Pocket TTS model moved to CUDA.")
        print("Pocket TTS model loaded.")
    return tts_model


def initialize_voice(reference_voice_path):
    global voice_state
    model = load_model()
    if not os.path.exists(reference_voice_path):
        raise FileNotFoundError(f"Reference voice file not found at '{reference_voice_path}'")
    print(f"Cloning voice from '{reference_voice_path}'...")
    voice_state = model.get_state_for_audio_prompt(reference_voice_path, truncate=True)
    print("Voice cloned successfully.")


@app.route('/tts', methods=['POST', 'PUT', 'GET'])
def tts_endpoint():
    print("\n--- New Pocket TTS Request Received ---")

    text_to_speak = None
    if request.method == 'GET':
        text_to_speak = request.args.get('text', '')
    else:
        data = request.get_json(silent=True)
        if data:
            text_to_speak = data.get('chatmessage') or data.get('text')

    if not text_to_speak:
        return jsonify({"error": "Missing 'chatmessage' or 'text' field"}), 400

    print(f"Synthesizing: '{text_to_speak[:100]}...'")

    try:
        model = load_model()
        audio = model.generate_audio(voice_state, text_to_speak)

        # audio shape is [channels, samples]; convert to numpy
        audio_np = audio.detach().cpu().numpy()
        if audio_np.ndim == 2 and audio_np.shape[0] == 1:
            audio_np = audio_np[0]
        sample_rate = model.sample_rate

        # Save to centralized TTS output folder for watcher_to_face.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tts_output_folder = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "tts_output")
        os.makedirs(tts_output_folder, exist_ok=True)
        output_filepath = os.path.join(tts_output_folder, "server_output.wav")

        scipy.io.wavfile.write(output_filepath, sample_rate, audio_np)
        print(f"Saved Pocket TTS audio to: {output_filepath}")

        # Return audio in response as well
        buffer = io.BytesIO()
        scipy.io.wavfile.write(buffer, sample_rate, audio_np)
        buffer.seek(0)
        return Response(buffer, mimetype='audio/wav')

    except Exception as e:
        print(f"Error during synthesis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error during audio generation"}), 500


@app.route('/', methods=['GET'])
def root():
    return jsonify({"status": "ok", "model": "pocket-tts"})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": "pocket-tts"})


@app.route('/settings', methods=['GET'])
def get_settings():
    return jsonify({
        'reference_voice': REFERENCE_VOICE,
        'device': DEVICE,
        'host': SERVER_HOST,
        'port': SERVER_PORT,
    })


if __name__ == "__main__":
    initialize_voice(reference_voice_path=REFERENCE_VOICE)
    print(f"\n--- Starting Pocket TTS Server on http://{SERVER_HOST}:{SERVER_PORT} ---")
    print(f" -> Endpoint: /tts (expects JSON {{'chatmessage': '...'}})")
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=SERVER_DEBUG)
