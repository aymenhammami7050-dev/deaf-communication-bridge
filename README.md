# Deaf Communication Bridge

Real-time two-way communication between deaf and hearing people.

- **Deaf → hearing:** hand gestures are detected from the webcam (OpenCV + MediaPipe), translated into a phrase, shown on screen and spoken aloud (text-to-speech).
- **Hearing → deaf:** speech is captured from the microphone, transcribed to text and displayed on screen. If no microphone is available, the app falls back to typing in the console.

Supports **English** and **French** (UI, spoken phrases and speech recognition).

## Features

- 7 single-hand gestures and 20+ two-hand combinations mapped to useful phrases
- Gesture smoothing and a "hold" timer (~0.4 s) to avoid false triggers
- Hand landmark overlay and hold-progress bar
- Speech-to-text via Google Speech Recognition (Whisper `tiny` as offline fallback)
- Microphone diagnostics and automatic noise-level calibration
- Typing fallback when no microphone is found
- Language selection screen at startup (EN / FR)

## Gestures

| Gesture (one hand) | Phrase |
|---|---|
| Open palm | I am happy |
| Closed fist | How are you |
| Pointing | You |
| Peace sign | Thank you |
| Thumbs up | Yes |
| OK sign | OK |
| Love sign (thumb + index + pinky) | I love you |

Two-hand combinations (e.g. open + open = "Hello!", fist + fist = "Congratulations", thumb + OK = "That is perfect") are defined in the `CombinedGesture` enum in `finale.py`.

## Requirements

- Python 3.9+
- A webcam
- A microphone (optional — typing fallback is available)

```bash
pip install -r requirements.txt
```

## Run

```bash
python finale.py
```

On first launch the MediaPipe `hand_landmarker.task` model is downloaded automatically.

Controls: `Q` = quit, `C` = clear text.

## How it works

1. **Capture:** OpenCV reads webcam frames.
2. **Detect:** MediaPipe Hand Landmarker returns 21 landmarks per hand.
3. **Classify:** finger up/down states and thumb–index distance are used to classify each hand, then a majority vote over the last frames smooths the result.
4. **Combine:** left/right gestures are looked up in a combination table.
5. **Speak:** after the gesture is held for 12 frames, the phrase is queued to a text-to-speech worker thread.
6. **Listen:** a background thread detects speech by volume threshold, then sends audio to Google Speech Recognition and displays the result.

## Platform notes

Text-to-speech currently uses Windows SAPI (`pywin32`) with a PowerShell fallback, so spoken output works on **Windows**. On Linux / Raspberry Pi, replace `_speak_text()` with a call to `espeak-ng` or `pyttsx3`; everything else (detection, speech-to-text, display) is cross-platform.

## Project structure

```
.
├── finale.py          # main application
├── requirements.txt
├── README.md
└── .gitignore
```

## Possible improvements

- Cross-platform TTS (espeak-ng / pyttsx3)
- Trained gesture classifier instead of rule-based finger states
- Offline speech recognition (Vosk / Whisper) as the default
- Support for more languages (Arabic)

## License

MIT
