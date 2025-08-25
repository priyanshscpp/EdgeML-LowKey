## Edge ML based Real-Time Keyword Spotting System

### Project Overview
Production-grade, real-time keyword spotting optimized for microcontrollers (ESP32-class) and validated on desktop with TensorFlow Lite. Detects: "all", "must", "none", "never", "only" (+ silence, unknown).

ESP32 (if you want to run on ESP32)
Replace the audio provider with an I2S-based implementation (16 kHz, mono, 16-bit)
Keep ring buffer size ≥ 16× DMA frame and set appropriate FreeRTOS task priorities.

### Features
- Robust streaming audio pipeline with wrap-safe ring buffer
- Efficient feature pipeline and memcpy-based shifting
- TFLM inference with compact op resolver
- Deterministic command smoothing/suppression
- Python accuracy and latency harnesses

### Tech Stack
- C/C++, FreeRTOS concepts, TFLite Micro
- Reference board: Arduino Nano 33 BLE Sense; adaptable to ESP32 (I2S)
- Python 3.10+ for desktop validation

### Setup (Desktop)
```
pip install -r scripts/requirements.txt
python scripts/evaluate_accuracy.py --model models/float_model.tflite --data_dir testing_audios
python scripts/benchmark_latency.py --model models/float_model.tflite
```

### Build (Arduino reference)
- Open `micro_speech/micro_speech.ino` in Arduino IDE with Arduino_TensorFlowLite installed. Flash to Nano 33 BLE Sense.

ESP32 notes:
- Replace `micro_speech/arduino_audio_provider.cpp` with an I2S provider (16 kHz, mono, 16-bit) and keep ring buffer sizing equal or larger.

### Usage
Device continuously listens; detections print over serial. LED toggles each inference on reference board.

### Performance
- Target on-device accuracy: ~95% for trained keywords
- Desktop latency stats printed by `benchmark_latency.py`

### Hexagon DSP Tips
- Align tensors (16B), avoid float, coalesce copies, use int8 kernels, and batch windowed FFTs where applicable.

### License
Apache 2.0
