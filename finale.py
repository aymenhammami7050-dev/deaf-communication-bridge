"""
Deaf Communication Bridge v2
Two-way communication between deaf and hearing people
With microphone diagnostics and typing fallback
"""

import cv2
import os
import math
import time
import urllib.request
import threading
import sys
import queue
import subprocess
import numpy as np
from enum import Enum
from collections import deque

# ============ GESTURE DETECTION ============

class SingleGesture(Enum):
    NONE = "none"
    OPEN_PALM = "I am happy"
    CLOSED_FIST = "How are you"
    POINTING = "You"
    PEACE_SIGN = "Thank you"
    THUMBS_UP = "Yes"
    OK_SIGN = "OK"
    LOVE_SIGN = "I love you"

class CombinedGesture(Enum):
    NONE = "none"
    OPEN_OPEN = "Hello!"
    FIST_FIST = "Congratulations"
    POINT_POINT = "Look over there"
    PEACE_PEACE = "All good between us"
    THUMB_THUMB = "Great job"
    OK_OK = "I am good"
    OPEN_FIST = "Tell me"
    FIST_OPEN = "Tell me"
    POINT_PEACE = "I understand you"
    PEACE_POINT = "I understand you"
    THUMB_FIST = "Keep going, you got this"
    FIST_THUMB = "Keep going, you got this"
    OK_POINT = "Ready the weekend"
    POINT_OK = "Ready the weekend"
    OPEN_PEACE = "I am free to talk"
    PEACE_OPEN = "I am free to talk"
    THUMB_PEACE = "That is amazing"
    PEACE_THUMB = "That is amazing"
    THUMB_OK = "That is perfect"
    OK_THUMB = "That is perfect"
    OPEN_OK = "I am ready"
    OK_OPEN = "I am ready"
    FIST_PEACE = "Processing"
    PEACE_FIST = "Processing"
    FIST_OK = "Deal, let's do it"
    OK_FIST = "Deal, let's do it"
    POINT_THUMB = "Good point"
    THUMB_POINT = "Good point"
    OPEN_THUMB = "Cheers, thank you"
    THUMB_OPEN = "Cheers, thank you"
    POINT_FIST = "Pay attention please"
    FIST_POINT = "Pay attention please"
    OK_PEACE = "I got this"
    PEACE_OK = "I got this"

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

SINGLE_GESTURE_TRANSLATIONS_FR = {
    "I am happy": "Je suis heureux",
    "How are you": "Comment allez-vous",
    "You": "Vous",
    "Thank you": "Merci",
    "Yes": "Oui",
    "OK": "D'accord",
    "I love you": "Je t'aime",
    "none": "aucun"
}

COMBINED_GESTURE_TRANSLATIONS_FR = {
    "Hello!": "Bonjour!",
    "Congratulations": "Félicitations",
    "Look over there": "Regardez là-bas",
    "All good between us": "Tout va bien entre nous",
    "Great job": "Excellent travail",
    "I am good": "Je vais bien",
    "Tell me": "Dites-moi",
    "I understand you": "Je vous comprends",
    "Keep going, you got this": "Continuez, vous y êtes presque",
    "Ready the weekend": "Prêt pour le week-end",
    "I am free to talk": "Je suis libre de parler",
    "That is amazing": "C'est incroyable",
    "That is perfect": "C'est parfait",
    "I am ready": "Je suis prêt",
    "Processing": "En cours",
    "Deal, let's do it": "Marché conclu",
    "Good point": "Bonne remarque",
    "Cheers, thank you": "Santé, merci",
    "Pay attention please": "Faites attention s'il vous plaît",
    "I got this": "J'ai compris"
}

LANGUAGE_CODES = {
    "en": "en-US",
    "fr": "fr-FR"
}


def prompt_language():
    width, height = 500, 280
    selected = [None]

    def draw_screen(hover=None):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)
        cv2.putText(img, "Deaf Communication Bridge", (60, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, "Choose your language / Choisissez votre langue", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        en_color = (0, 200, 100) if hover == "en" else (0, 150, 70)
        cv2.rectangle(img, (50, 120), (210, 210), en_color, -1)
        cv2.rectangle(img, (50, 120), (210, 210), (255, 255, 255), 2)
        cv2.putText(img, "EN", (95, 178), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
        cv2.putText(img, "English", (68, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        fr_color = (200, 80, 0) if hover == "fr" else (150, 60, 0)
        cv2.rectangle(img, (290, 120), (450, 210), fr_color, -1)
        cv2.rectangle(img, (290, 120), (450, 210), (255, 255, 255), 2)
        cv2.putText(img, "FR", (328, 178), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
        cv2.putText(img, "Francais", (305, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        return img

    def mouse_callback(event, x, y, flags, param):
        hover = None
        if 50 <= x <= 210 and 120 <= y <= 210:
            hover = "en"
        elif 290 <= x <= 450 and 120 <= y <= 210:
            hover = "fr"
        if event == cv2.EVENT_LBUTTONDOWN:
            if hover:
                selected[0] = hover
        else:
            img = draw_screen(hover)
            cv2.imshow("Language / Langue", img)

    cv2.namedWindow("Language / Langue")
    cv2.setMouseCallback("Language / Langue", mouse_callback)
    cv2.imshow("Language / Langue", draw_screen())

    while selected[0] is None:
        cv2.imshow("Language / Langue", draw_screen())
        cv2.waitKey(30)

    cv2.destroyWindow("Language / Langue")
    return selected[0]


def translate_gesture_text(gesture_text, language="en"):
    if language == "fr":
        return SINGLE_GESTURE_TRANSLATIONS_FR.get(
            gesture_text, COMBINED_GESTURE_TRANSLATIONS_FR.get(gesture_text, gesture_text))
    return gesture_text


def _sapi_speak(text, language):
    try:
        import win32com.client
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        lang_code = "0409" if language == "en" else "040c"
        voices = speaker.GetVoices()
        for i in range(voices.Count):
            voice = voices.Item(i)
            voice_lang = str(voice.GetAttribute("Language")) if hasattr(voice, 'GetAttribute') else str(voice.Language)
            if lang_code in voice_lang:
                speaker.Voice = voice
                break
        speaker.Speak(text)
        return True
    except Exception as e:
        print(f"[TTS] SAPI error: {e}")
        return False


def _powershell_speak(text, language):
    try:
        import base64
        encoded = base64.b64encode(text.encode('utf-8')).decode('ascii')
        lang_code = LANGUAGE_CODES.get(language, 'en-US')
        ps = (
            "$text = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String('" + encoded + "')); "
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$voice = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -eq '" + lang_code + "' } | Select-Object -First 1; "
            "if ($voice) { $s.SelectVoice($voice.VoiceInfo.Name) }; "
            "$s.Speak($text);"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True, timeout=12)
        return True
    except Exception as e:
        print(f"[TTS] PowerShell error: {e}")
        return False


def _speak_text(text, language):
    if not text:
        return
    if _sapi_speak(text, language):
        return
    _powershell_speak(text, language)


def speech_worker(speech_queue, language):
    while True:
        text = speech_queue.get()
        if text is None:
            break
        _speak_text(text, language)


def enqueue_speech(text, speech_queue, last_text, last_time, cooldown=1.5):
    now = time.time()
    if not text or text == "none":
        return last_text, last_time
    if text == last_text:
        return last_text, last_time
    speech_queue.put(text)
    return text, now


def download_model():
    if not os.path.exists(MODEL_PATH):
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Model downloaded!")
        except:
            pass


def create_landmarker():
    import mediapipe as mp
    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        return HandLandmarker.create_from_options(options), "tasks"
    except:
        pass
    try:
        hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        return hands, "legacy"
    except:
        pass
    return None, None


class HandTracker:
    def __init__(self):
        download_model()
        self.landmarker, self.api_type = create_landmarker()
        if not self.landmarker:
            print("ERROR: No hand detection!")
            exit(1)
        self.left_history = deque(maxlen=5)
        self.right_history = deque(maxlen=5)
        self.combined_history = deque(maxlen=5)

    def detect(self, frame):
        frame = cv2.flip(frame, 1)
        if self.api_type == "tasks":
            return self._detect_tasks(frame)
        return self._detect_legacy(frame)

    def _detect_tasks(self, frame):
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.landmarker.detect_for_video(mp_image, int(time.time() * 1000))
        hands_data = []
        if results.hand_landmarks:
            for i, landmarks in enumerate(results.hand_landmarks):
                label = "Right" if i == 0 else "Left"
                if results.handedness and i < len(results.handedness):
                    label = results.handedness[i][0].display_name
                info = self._process(landmarks, frame.shape, label)
                hands_data.append(info)
                self._draw(frame, landmarks)
        return hands_data, frame

    def _detect_legacy(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.landmarker.process(rgb)
        hands_data = []
        if results.multi_hand_landmarks:
            for hl, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                info = self._process(hl.landmark, frame.shape, hd.classification[0].label)
                hands_data.append(info)
                self._draw_legacy(frame, hl)
        return hands_data, frame

    def _process(self, landmarks, shape, label):
        h, w, _ = shape
        pts = [{"x": int(lm.x * w), "y": int(lm.y * h)} for lm in landmarks]
        fingers = self._finger_states(pts)
        gesture = self._classify_single(fingers, pts)
        if label == "Left":
            self.left_history.append(gesture)
        else:
            self.right_history.append(gesture)
        smoothed = self._smooth(label)
        return {"label": label, "gesture": smoothed, "fingers": sum(fingers)}

    def _finger_states(self, pts):
        thumb = abs(pts[4]["x"] - pts[0]["x"]) > abs(pts[3]["x"] - pts[0]["x"]) * 1.1
        index = pts[8]["y"] < pts[6]["y"]
        middle = pts[12]["y"] < pts[10]["y"]
        ring = pts[16]["y"] < pts[14]["y"]
        pinky = pts[20]["y"] < pts[18]["y"]
        return [thumb, index, middle, ring, pinky]

    def _classify_single(self, fingers, pts):
        thumb, index, middle, ring, pinky = fingers
        dist = math.hypot(pts[4]["x"] - pts[8]["x"], pts[4]["y"] - pts[8]["y"])
        if dist < 35 and (middle or ring or pinky):
            return SingleGesture.OK_SIGN
        if thumb and index and not middle and not ring and pinky:
            return SingleGesture.LOVE_SIGN
        if thumb and not index and not middle and not ring and not pinky:
            if pts[4]["y"] < pts[3]["y"]:
                return SingleGesture.THUMBS_UP
        if not thumb and index and middle and not ring and not pinky:
            return SingleGesture.PEACE_SIGN
        if not thumb and index and not middle and not ring and not pinky:
            return SingleGesture.POINTING
        if sum(fingers) >= 4:
            return SingleGesture.OPEN_PALM
        if sum(fingers) == 0:
            return SingleGesture.CLOSED_FIST
        return SingleGesture.NONE

    def _smooth(self, label):
        from collections import Counter
        hist = self.left_history if label == "Left" else self.right_history
        if len(hist) < 3:
            return SingleGesture.NONE
        c = Counter(hist)
        m = c.most_common(1)[0]
        return m[0] if m[1] >= 3 else SingleGesture.NONE

    def get_display_gesture(self, hands_data):
        if len(hands_data) == 0:
            return "none", "single"
        if len(hands_data) == 1:
            return hands_data[0]["gesture"].value, "single"
        left_g = right_g = None
        for h in hands_data:
            if h["label"] == "Left":
                left_g = h["gesture"]
            else:
                right_g = h["gesture"]
        if not left_g or not right_g:
            left_g = hands_data[0]["gesture"]
            right_g = hands_data[1]["gesture"]
        if left_g == SingleGesture.NONE or right_g == SingleGesture.NONE:
            active = left_g if right_g == SingleGesture.NONE else right_g
            return active.value, "single"
        combo_map = {
            (SingleGesture.OPEN_PALM, SingleGesture.OPEN_PALM): CombinedGesture.OPEN_OPEN,
            (SingleGesture.CLOSED_FIST, SingleGesture.CLOSED_FIST): CombinedGesture.FIST_FIST,
            (SingleGesture.POINTING, SingleGesture.POINTING): CombinedGesture.POINT_POINT,
            (SingleGesture.PEACE_SIGN, SingleGesture.PEACE_SIGN): CombinedGesture.PEACE_PEACE,
            (SingleGesture.THUMBS_UP, SingleGesture.THUMBS_UP): CombinedGesture.THUMB_THUMB,
            (SingleGesture.OK_SIGN, SingleGesture.OK_SIGN): CombinedGesture.OK_OK,
            (SingleGesture.OPEN_PALM, SingleGesture.CLOSED_FIST): CombinedGesture.OPEN_FIST,
            (SingleGesture.CLOSED_FIST, SingleGesture.OPEN_PALM): CombinedGesture.FIST_OPEN,
            (SingleGesture.POINTING, SingleGesture.PEACE_SIGN): CombinedGesture.POINT_PEACE,
            (SingleGesture.PEACE_SIGN, SingleGesture.POINTING): CombinedGesture.PEACE_POINT,
            (SingleGesture.THUMBS_UP, SingleGesture.CLOSED_FIST): CombinedGesture.THUMB_FIST,
            (SingleGesture.CLOSED_FIST, SingleGesture.THUMBS_UP): CombinedGesture.FIST_THUMB,
            (SingleGesture.OK_SIGN, SingleGesture.POINTING): CombinedGesture.OK_POINT,
            (SingleGesture.POINTING, SingleGesture.OK_SIGN): CombinedGesture.POINT_OK,
            (SingleGesture.OPEN_PALM, SingleGesture.PEACE_SIGN): CombinedGesture.OPEN_PEACE,
            (SingleGesture.PEACE_SIGN, SingleGesture.OPEN_PALM): CombinedGesture.PEACE_OPEN,
            (SingleGesture.THUMBS_UP, SingleGesture.PEACE_SIGN): CombinedGesture.THUMB_PEACE,
            (SingleGesture.PEACE_SIGN, SingleGesture.THUMBS_UP): CombinedGesture.PEACE_THUMB,
            (SingleGesture.THUMBS_UP, SingleGesture.OK_SIGN): CombinedGesture.THUMB_OK,
            (SingleGesture.OK_SIGN, SingleGesture.THUMBS_UP): CombinedGesture.OK_THUMB,
            (SingleGesture.OPEN_PALM, SingleGesture.OK_SIGN): CombinedGesture.OPEN_OK,
            (SingleGesture.OK_SIGN, SingleGesture.OPEN_PALM): CombinedGesture.OK_OPEN,
            (SingleGesture.CLOSED_FIST, SingleGesture.PEACE_SIGN): CombinedGesture.FIST_PEACE,
            (SingleGesture.PEACE_SIGN, SingleGesture.CLOSED_FIST): CombinedGesture.PEACE_FIST,
            (SingleGesture.CLOSED_FIST, SingleGesture.OK_SIGN): CombinedGesture.FIST_OK,
            (SingleGesture.OK_SIGN, SingleGesture.CLOSED_FIST): CombinedGesture.OK_FIST,
            (SingleGesture.POINTING, SingleGesture.THUMBS_UP): CombinedGesture.POINT_THUMB,
            (SingleGesture.THUMBS_UP, SingleGesture.POINTING): CombinedGesture.THUMB_POINT,
            (SingleGesture.OPEN_PALM, SingleGesture.THUMBS_UP): CombinedGesture.OPEN_THUMB,
            (SingleGesture.THUMBS_UP, SingleGesture.OPEN_PALM): CombinedGesture.THUMB_OPEN,
            (SingleGesture.POINTING, SingleGesture.CLOSED_FIST): CombinedGesture.POINT_FIST,
            (SingleGesture.CLOSED_FIST, SingleGesture.POINTING): CombinedGesture.FIST_POINT,
            (SingleGesture.OK_SIGN, SingleGesture.PEACE_SIGN): CombinedGesture.OK_PEACE,
            (SingleGesture.PEACE_SIGN, SingleGesture.OK_SIGN): CombinedGesture.PEACE_OK,
        }
        combined = combo_map.get((left_g, right_g), CombinedGesture.NONE)
        self.combined_history.append(combined)
        from collections import Counter
        if len(self.combined_history) >= 3:
            c = Counter(self.combined_history)
            m = c.most_common(1)[0]
            if m[1] >= 2:
                combined = m[0]
        return combined.value, "combined"

    def _draw(self, frame, landmarks):
        h, w, _ = frame.shape
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        conns = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                 (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                 (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
        for s, e in conns:
            cv2.line(frame, pts[s], pts[e], (0, 255, 0), 2)
        for i, p in enumerate(pts):
            sz = 8 if i in [4,8,12,16,20] else 4
            col = (0,0,255) if i in [4,8,12,16,20] else (255,255,0)
            cv2.circle(frame, p, sz, col, -1)

    def _draw_legacy(self, frame, hl):
        h, w, _ = frame.shape
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hl.landmark]
        conns = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                 (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                 (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
        for s, e in conns:
            cv2.line(frame, pts[s], pts[e], (0, 255, 0), 2)
        for i, p in enumerate(pts):
            sz = 8 if i in [4,8,12,16,20] else 4
            col = (0,0,255) if i in [4,8,12,16,20] else (255,255,0)
            cv2.circle(frame, p, sz, col, -1)

    def release(self):
        if self.landmarker:
            self.landmarker.close()


# ============ SPEECH TO TEXT ============

class SpeechToText:
    def __init__(self, language="en-US"):
        self.available = False
        self.error_msg = ""
        self.mic_name = "None"
        self.language = language
        self.text_lines = deque(maxlen=10)
        self.current_text = ""
        self.is_listening = False
        self._listen_thread = None
        self._diagnostics = []
        self._sample_rate = 16000
        self._last_volume = 0
        self._last_threshold = 100
        self.is_processing = False
        self._clear_timer = None

        try:
            import sounddevice as sd
            self.sd = sd
            self._add_diag("sounddevice: OK")
        except ImportError:
            self.error_msg = "Install: pip install sounddevice"
            self._add_diag("sounddevice: MISSING")
            return

        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            if len(input_devices) == 0:
                self.error_msg = "No microphone found!"
                self._add_diag("Microphone: NONE FOUND")
                return
            default_mic = sd.query_devices(kind='input')
            self.mic_name = default_mic['name']
            self._sample_rate = int(default_mic['default_samplerate'])
            self._add_diag(f"Found {len(input_devices)} mic(s)")
            self._add_diag(f"Using: {self.mic_name}")
            self._add_diag(f"Sample rate: {self._sample_rate}")
            self.available = True
        except Exception as e:
            self.error_msg = f"Mic error: {str(e)[:50]}"
            self._add_diag(f"Mic error: {str(e)[:60]}")

    def _add_diag(self, msg):
        self._diagnostics.append(msg)
        print(f"[Speech] {msg}")

    def _clear_heard(self):
        self.current_text = ""
        self._clear_timer = None

    def _schedule_clear(self, delay=5.0):
        if self._clear_timer:
            self._clear_timer.cancel()
        self._clear_timer = threading.Timer(delay, self._clear_heard)
        self._clear_timer.start()

    def start_listening(self):
        if not self.available:
            self._add_diag("Cannot start - mic not available")
            return
        self.is_listening = True
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()
        self._add_diag("Listening started!")

    def stop_listening(self):
        self.is_listening = False
        if self._clear_timer:
            self._clear_timer.cancel()

    def _audio_to_wav_bytes(self, audio_data):
        import io
        import wave
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(audio_data.tobytes())
        return buf.getvalue()

    def _listen_loop(self):
        import numpy as np

        block_duration = 0.2
        block_samples = int(self._sample_rate * block_duration)
        silence_limit = 0.8
        min_speech_duration = 0.3
        max_speech_duration = 8.0

        audio_buffer = []
        speech_audio = []
        silence_blocks = 0
        is_speaking = False

        self._add_diag("Calibrating noise level...")
        try:
            noise_sample = self.sd.rec(int(self._sample_rate * 0.5),
                                       samplerate=self._sample_rate, channels=1, dtype='int16')
            self.sd.wait()
            noise_level = np.abs(noise_sample).mean()
            if noise_level == 0:
                self.error_msg = "Microphone not recording! Check Windows mic settings"
                self._add_diag("ERROR: Microphone returns silence (level=0)")
                self._add_diag("Fix: Windows Settings > Sound > Select correct input")
                self.available = False
                return
            volume_threshold = max(noise_level * 1.5, 35)
            self._add_diag(f"Noise level: {noise_level:.0f} | Threshold: {volume_threshold:.0f}")
        except:
            volume_threshold = 50
            self._add_diag(f"Using default threshold: {volume_threshold}")

        self._add_diag("Speak clearly near the microphone!")
        self._add_diag("Say a complete phrase and pause - it will process automatically.")

        def callback(indata, frames, time_info, status):
            audio_buffer.append(indata.copy())

        with self.sd.InputStream(samplerate=self._sample_rate, channels=1,
                                 dtype='int16', blocksize=block_samples,
                                 callback=callback):
            while self.is_listening:
                time.sleep(block_duration * 0.7)
                if not audio_buffer:
                    continue

                new_blocks = len(audio_buffer)
                recent = np.concatenate(audio_buffer)
                audio_buffer.clear()

                volume = np.abs(recent).mean()
                self._last_volume = volume
                self._last_threshold = volume_threshold

                if volume > volume_threshold:
                    speech_audio.append(recent)
                    silence_blocks = 0
                    if not is_speaking:
                        is_speaking = True
                        self._add_diag("Hearing speech...")
                else:
                    if is_speaking:
                        silence_blocks += new_blocks
                        if silence_blocks * block_duration <= silence_limit * 0.5:
                            speech_audio.append(recent)
                        if silence_blocks * block_duration > silence_limit:
                            is_speaking = False
                            if len(speech_audio) > 0:
                                total_audio = np.concatenate(speech_audio)
                                total_duration = len(total_audio) / self._sample_rate
                                if total_duration >= min_speech_duration:
                                    self._add_diag(f"Processing {total_duration:.1f}s of speech...")
                                    self._process_audio(total_audio)
                                else:
                                    self._add_diag(f"Speech too short ({total_duration:.1f}s), ignored")
                            speech_audio = []
                            silence_blocks = 0

                if is_speaking and len(speech_audio) > 0:
                    total_samples = sum(len(chunk) for chunk in speech_audio)
                    total_duration = total_samples / self._sample_rate
                    if total_duration >= max_speech_duration:
                        self._add_diag(f"Processing long speech ({total_duration:.1f}s)...")
                        total_audio = np.concatenate(speech_audio)
                        self._process_audio(total_audio)
                        speech_audio = []
                        is_speaking = False
                        silence_blocks = 0

    def _process_audio(self, audio_data):
        t = threading.Thread(target=self._do_process, args=(audio_data,), daemon=True)
        t.start()

    def _do_process(self, audio_data):
        import numpy as np
        self.is_processing = True
        try:
            wav_bytes = self._audio_to_wav_bytes(audio_data.flatten())
            try:
                import speech_recognition as sr
            except ImportError:
                self._add_diag("speech_recognition not available")
                self._try_whisper(audio_data.flatten())
                return

            try:
                recognizer = sr.Recognizer()
                audio = sr.AudioData(wav_bytes, self._sample_rate, 2)
                text = recognizer.recognize_google(audio, language=self.language)
                if text and text.strip():
                    self.current_text = text.strip()
                    self.text_lines.append({"text": text.strip(), "time": time.strftime("%H:%M")})
                    self._add_diag(f"Google: {text.strip()}")
                    self._schedule_clear(5.0)
                    return
            except sr.UnknownValueError:
                self._add_diag("Google: Could not understand")
                return
            except sr.RequestError as e:
                self._add_diag(f"Google API error: {str(e)[:40]}")
                return
            except Exception as e:
                self._add_diag(f"Google error: {type(e).__name__}")
                return
        except Exception as e:
            self._add_diag(f"Process error: {str(e)[:40]}")
        finally:
            self.is_processing = False

    def _try_whisper(self, audio_data):
        self._add_diag("WARNING: Using Whisper (slow). Check internet for Google API.")
        try:
            import whisper
            if not hasattr(self, '_whisper_model'):
                self._add_diag("Loading Whisper tiny model...")
                self._whisper_model = whisper.load_model("tiny")
                self._add_diag("Whisper model loaded!")

            import tempfile
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                sf.write(f.name, audio_data, self._sample_rate)
                result = self._whisper_model.transcribe(f.name, language="en")
                text = result.get("text", "").strip()
                if text and text not in ["you", "", "."]:
                    self.current_text = text
                    self.text_lines.append({"text": text, "time": time.strftime("%H:%M")})
                    self._add_diag(f"Whisper: {text}")
                    self._schedule_clear(5.0)
                else:
                    self._add_diag("Whisper: No speech detected")
                os.unlink(f.name)
        except ImportError:
            self._add_diag("Whisper not installed")
        except Exception as e:
            self._add_diag(f"Whisper error: {type(e).__name__}: {str(e)[:50]}")

    def get_recent_text(self, count=5):
        return list(self.text_lines)[-count:]

    def get_current(self):
        return self.current_text

    def get_diagnostics(self):
        return list(self._diagnostics)[-10:]

    def clear(self):
        self.text_lines.clear()
        self.current_text = ""
        if self._clear_timer:
            self._clear_timer.cancel()
            self._clear_timer = None


# ============ TEXT INPUT FALLBACK ============

class TextInputFallback:
    def __init__(self):
        self.text_queue = deque(maxlen=20)
        self._input_thread = None
        self._running = False

    def start(self):
        self._running = True
        self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._input_thread.start()

    def stop(self):
        self._running = False

    def _input_loop(self):
        print("\n" + "="*60)
        print("TYPE TEXT INPUT ACTIVE")
        print("Type what the hearing person says and press ENTER")
        print("Type 'quit' to exit")
        print("="*60 + "\n")
        while self._running:
            try:
                text = input("Hearing person says: ")
                if text.lower() == 'quit':
                    break
                if text.strip():
                    self.text_queue.append({
                        "text": text.strip(),
                        "time": time.strftime("%H:%M")
                    })
                    print(f"  -> Added: {text.strip()}")
            except EOFError:
                break
            except Exception:
                time.sleep(0.5)

    def get_recent_text(self, count=5):
        return list(self.text_queue)[-count:]

    def get_current(self):
        return self.text_queue[-1]["text"] if self.text_queue else ""

    def clear(self):
        self.text_queue.clear()


# ============ MAIN APPLICATION ============

def draw_text_box(frame, text_lines, x, y, w, h, title="Speaker", border_color=(255, 255, 255)):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (x, y), (x+w, y+h), border_color, 2)
    cv2.rectangle(frame, (x, y), (x+w, y+30), (200, 100, 0), -1)
    cv2.putText(frame, f"  {title}", (x, y+22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    line_y = y + 50
    for line_data in text_lines:
        text = line_data.get("text", line_data) if isinstance(line_data, dict) else line_data
        words = str(text).split()
        current = ""
        for word in words:
            if len(current) + len(word) > 35:
                cv2.putText(frame, current, (x+10, line_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                line_y += 25
                current = word
            else:
                current = current + " " + word if current else word
        if current:
            cv2.putText(frame, current, (x+10, line_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            line_y += 25
        if line_y > y + h - 20:
            break


def main():
    print("="*70)
    print("  DEAF COMMUNICATION BRIDGE v2")
    print("="*70)

    cap = cv2.VideoCapture(0)
    hand_tracker = HandTracker()
    speech = SpeechToText()

    use_typing = not speech.available
    text_input = None

    if use_typing:
        print("\n⚠️ Microphone not available!")
        print(f"   Reason: {speech.error_msg}")
        print("\n✓ Starting TEXT TYPING fallback")
        text_input = TextInputFallback()
        text_input.start()
    else:
        print("\n✓ Microphone found!")
        print(f"   Device: {speech.mic_name}")
        speech.start_listening()
        text_input = speech

    language = prompt_language()
    speech.language = LANGUAGE_CODES.get(language, speech.language)
    print(f"\n✓ Voice language set to: {language}")

    speech_queue = queue.Queue()
    speech_thread = threading.Thread(target=speech_worker, args=(speech_queue, language), daemon=True)
    speech_thread.start()

    last_spoken_gesture = None
    last_spoken_time = 0.0
    last_displayed_gesture = "none"
    none_frames = 0
    gesture_hold_frames = 0
    required_hold = 12       # frames gesture must be stable before triggering (~0.4s)
    pending_gesture = "none"

    print("\nControls:")
    print("  Q = Quit | C = Clear all text")
    print("="*70)

    gesture_history = deque(maxlen=5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hands_data, display = hand_tracker.detect(frame)
        gesture_text, mode = hand_tracker.get_display_gesture(hands_data)
        translated_gesture = translate_gesture_text(gesture_text, language)

        if gesture_text != "none":
            none_frames = 0

            if translated_gesture == pending_gesture:
                gesture_hold_frames += 1
            else:
                # Gesture changed mid-transition — reset hold counter
                pending_gesture = translated_gesture
                gesture_hold_frames = 0

            # Only trigger once gesture has been held long enough
            if gesture_hold_frames == required_hold and translated_gesture != last_displayed_gesture:
                while not speech_queue.empty():
                    try:
                        speech_queue.get_nowait()
                    except:
                        pass
                last_spoken_gesture, last_spoken_time = enqueue_speech(
                    translated_gesture, speech_queue,
                    last_spoken_gesture, last_spoken_time)
                last_displayed_gesture = translated_gesture
                gesture_history.append({"text": translated_gesture, "time": time.strftime("%H:%M")})
        else:
            none_frames += 1
            gesture_hold_frames = 0
            pending_gesture = "none"
            if none_frames >= 15:
                last_displayed_gesture = "none"

        h, w = display.shape[:2]

        # --- TOP: Gesture display ---
        if gesture_text != "none":
            color = (0, 255, 0) if mode == "single" else (0, 200, 255)
            # Show pending gesture with hold progress bar
            label = translated_gesture if gesture_hold_frames >= required_hold else f"{pending_gesture}..."
            cv2.putText(display, f"SIGN: {label.upper()}",
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            mode_label = "1 HAND" if mode == "single" else "2 HANDS"
            cv2.putText(display, mode_label, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            # Hold progress bar
            bar_fill = int((gesture_hold_frames / required_hold) * 150)
            bar_fill = min(bar_fill, 150)
            cv2.rectangle(display, (10, 80), (160, 90), (50, 50, 50), -1)
            cv2.rectangle(display, (10, 80), (10 + bar_fill, 90), (0, 220, 120), -1)
        else:
            no_gesture_prompt = "Montrez un geste" if language == "fr" else "Show sign language gesture"
            cv2.putText(display, no_gesture_prompt,
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)

        # --- MIC STATUS ---
        if speech.available:
            cv2.putText(display, "MIC: OK", (w-150, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            error_short = speech.error_msg[:20] if speech.error_msg else "NO MIC"
            cv2.putText(display, f"MIC: {error_short}", (w-250, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(display, "(Type in console)", (w-250, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        cv2.putText(display, f"Hands: {len(hands_data)}", (w-150, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # --- LISTENING INDICATOR ---
        if speech.available and hasattr(speech, '_last_volume'):
            vol = min(speech._last_volume, 200)
            bar_w = 100
            bar_x = w - 120
            bar_y = 65
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (50, 50, 50), -1)
            cv2.rectangle(display, (bar_x, bar_y),
                         (bar_x + int(bar_w * vol / 200), bar_y + 8), (0, 255, 0), -1)
            if speech._last_volume > speech._last_threshold * 0.8:
                blink = (int(time.time() * 3) % 2) == 0
                color = (0, 0, 255) if blink else (100, 100, 100)
                cv2.circle(display, (w - 135, 69), 5, color, -1)

        # --- PROCESSING INDICATOR ---
        if speech.available and speech.is_processing:
            blink = (int(time.time() * 4) % 2) == 0
            if blink:
                cv2.putText(display, "PROCESSING...", (w//2 - 100, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # --- SPEECH DISPLAY ---
        current_speech = text_input.get_current()
        if current_speech:
            cv2.rectangle(display, (50, 80), (w-50, 150), (0, 0, 80), -1)
            cv2.rectangle(display, (50, 80), (w-50, 150), (255, 255, 255), 2)
            display_text = current_speech if len(current_speech) < 45 else current_speech[:42] + "..."
            cv2.putText(display, display_text, (60, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            source = "TYPED" if use_typing else "HEARD"
            cv2.putText(display, source, (60, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # --- BOTTOM LEFT: Gesture history ---
        gesture_lines = list(gesture_history)[-5:]
        draw_text_box(display, gesture_lines, 10, h-220, w//2-20, 210,
                     "Deaf Person Signs", (0, 200, 255))

        # --- BOTTOM RIGHT: Speech text ---
        speech_lines = text_input.get_recent_text(5)
        draw_text_box(display, speech_lines, w//2+10, h-220, w//2-20, 210,
                     "Hearing Person Says", (0, 255, 100))

        # --- Bottom bar ---
        controls = "Q=Quit | C=Clear"
        if use_typing:
            controls += " | TYPE in console window"
        cv2.putText(display, controls, (10, h-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        cv2.imshow("Deaf Communication Bridge v2", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            text_input.clear()
            gesture_history.clear()

    if use_typing and text_input:
        text_input.stop()
    else:
        speech.stop_listening()

    speech_queue.put(None)
    speech_thread.join(timeout=3)

    cap.release()
    hand_tracker.release()
    cv2.destroyAllWindows()
    print("\nBridge closed!")


if __name__ == "__main__":
    main()