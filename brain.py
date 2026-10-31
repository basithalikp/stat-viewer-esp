import json
import os
import queue
import re
import subprocess
import time
import docker
import paho.mqtt.publish as publish
import requests
import sounddevice as sd
from vosk import KaldiRecognizer, Model

# --- CONFIGURATION ---
MQTT_BROKER = "127.0.0.1"
MQTT_TOPIC_STATS = "deskbuddy/stats"
MQTT_TOPIC_EVENT = "deskbuddy/event"

DESKTOP_OLLAMA = "http://192.168.1.150:11434"  # Replace with Desktop IP
CLOUD_OLLAMA = "http://your-remote-vps:11434"    # Fallback Cloud Ollama
OLLAMA_MODEL = "llama3.2:1b"

VOSK_MODEL_PATH = "/opt/deskbuddy/vosk-model-small-en-us-0.15"
PIPER_MODEL = "/opt/deskbuddy/en_US-lessac-medium.onnx"

audio_queue = queue.Queue()
docker_client = docker.from_env()

# --- AUDIO & TTS ENGINE ---
def audio_callback(indata, frames, time_info, status):
    audio_queue.put(bytes(indata))

def speak(text: str):
    """Speaks through the connected Bluetooth speaker using Piper TTS."""
    clean_text = re.sub(r'[*_#`]', '', text).strip()
    cmd = f'echo "{clean_text}" | piper --model {PIPER_MODEL} --output_raw | aplay -r 22050 -f S16_LE -t raw'
    subprocess.Popen(cmd, shell=True)

# --- DISPLAY PUBLISHER ---
def update_oled(data: dict):
    publish.single(MQTT_TOPIC_STATS, json.dumps(data), hostname=MQTT_BROKER)

# --- SYSTEM METRICS COLLECTOR ---
def get_system_metrics():
    # CPU Temp
    with open("/sys/class/thermal/thermal_zone0/temp") as f:
        cpu_temp = int(f.read().strip()) // 1000

    # RAM Usage
    with open('/proc/meminfo') as f:
        meminfo = f.read()
    total = int(re.search(r'MemTotal:\s+(\d+)', meminfo).group(1))
    avail = int(re.search(r'MemAvailable:\s+(\d+)', meminfo).group(1))
    ram_gb = round((total - avail) / 1024 / 1024, 1)

    # Active Containers
    containers = docker_client.containers.list()
    running_count = len(containers)

    return {
        "cpu_temp": cpu_temp,
        "ram_usage": ram_gb,
        "containers": f"{running_count} UP",
        "jellyfin": "IDLE",     # Wire to Jellyfin API
        "qbittorrent": "0 DL"   # Wire to qBittorrent API
    }

# --- HYBRID OLLAMA ROUTER ---
def query_ollama(prompt: str) -> str:
    """Checks Desktop Ollama first; falls back to Cloud Ollama."""
    target_url = CLOUD_OLLAMA
    try:
        # Fast 200ms ping to check if desktop is powered on
        r = requests.get(f"{DESKTOP_OLLAMA}/api/version", timeout=0.2)
        if r.status_code == 200:
            target_url = DESKTOP_OLLAMA
    except Exception:
        target_url = CLOUD_OLLAMA

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": "You are a concise desk voice assistant. Reply in max 1-2 brief sentences.",
        "stream": False
    }

    try:
        res = requests.post(f"{target_url}/api/generate", json=payload, timeout=6.0)
        return res.json().get("response", "I could not reach the intelligence backend.")
    except Exception:
        return "Both desktop and cloud reasoning backends are offline."

# --- INTENT DISPATCHER ---
def handle_command(text: str):
    publish.single(MQTT_TOPIC_EVENT, json.dumps({"status": "LISTENING", "text": text}), hostname=MQTT_BROKER)
    text = text.lower()

    if "temperature" in text or "cpu" in text:
        metrics = get_system_metrics()
        reply = f"The CPU is cruising at {metrics['cpu_temp']} degrees Celsius."
        speak(reply)
        return

    if "container" in text or "docker" in text:
        metrics = get_system_metrics()
        reply = f"All {metrics['containers']} containers are currently healthy."
        speak(reply)
        return

    if "download" in text or "torrent" in text:
        # Query your qBittorrent API here
        reply = "You currently have two active downloads at 3.4 megabytes per second."
        speak(reply)
        return

    # Fallback: Unrecognized command routed to Ollama
    answer = query_ollama(text)
    speak(answer)

# --- MAIN LOOP ---
def main():
    model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, 16000)

    last_stats_tick = 0

    with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype='int16', channels=1, callback=audio_callback):
        print("Desk Buddy Server Brain Initialized...")
        while True:
            # Broadcast idle system telemetry every 3 seconds
            if time.time() - last_stats_tick > 3:
                update_oled(get_system_metrics())
                last_stats_tick = time.time()

            # Process Speech
            try:
                data = audio_queue.get_nowait()
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    raw_text = res.get("text", "").strip()
                    if raw_text:
                        print(f"Recognized: {raw_text}")
                        handle_command(raw_text)
            except queue.Empty:
                time.sleep(0.05)

if __name__ == "__main__":
    main()