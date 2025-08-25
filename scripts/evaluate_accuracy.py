import argparse
import os
import glob
import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf


def load_wav_mono_16k(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype='int16', always_2d=False)
    if y.ndim > 1:
        y = y[:, 0]
    if sr != 16000:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000)
        y = y.astype(np.int16)
    return y


def frame_to_spectrogram(int16_pcm: np.ndarray) -> np.ndarray:
    # Mirror the training-time scaling from micro_features generator
    # Compute log-mel filterbank with 40 channels, 30ms window, 20ms hop
    y = int16_pcm.astype(np.float32) / 32768.0
    n_fft = 512
    hop_length = int(0.02 * 16000)
    win_length = int(0.03 * 16000)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=16000,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=40,
        fmin=125.0,
        fmax=7500.0,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=1.0)
    # Normalize approximately to match int8 quantized input distribution
    S_norm = (S_db - S_db.mean()) / (S_db.std() + 1e-6)
    # Quantize to int8-like range
    S_q = np.clip(np.round(S_norm * 10.0) + 0, -128, 127)
    return S_q.astype(np.int8)


def prepare_input_window(spec: np.ndarray) -> np.ndarray:
    # Expect 40 x T; take last 49 frames or pad
    num_frames = 49
    if spec.shape[1] < num_frames:
        pad = np.zeros((40, num_frames - spec.shape[1]), dtype=np.int8)
        spec = np.concatenate([spec, pad], axis=1)
    else:
        spec = spec[:, -num_frames:]
    flat = spec.T.reshape(1, 40 * num_frames)
    return flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data_dir', required=True)
    args = ap.parse_args()

    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    wavs = sorted(glob.glob(os.path.join(args.data_dir, '*.wav')))
    if not wavs:
        print('No wav files found')
        return

    labels = ['silence', 'unknown', 'all', 'must', 'none', 'never', 'only']
    correct = 0
    total = 0
    for wav in wavs:
        pcm = load_wav_mono_16k(wav)
        spec = frame_to_spectrogram(pcm)
        x = prepare_input_window(spec)
        interpreter.set_tensor(input_details['index'], x.astype(np.int8))
        interpreter.invoke()
        out = interpreter.get_tensor(output_details['index'])[0]
        pred = int(np.argmax(out))
        pred_label = labels[pred]
        base = os.path.basename(wav).split('.')[0]
        total += 1
        if base in pred_label:
            correct += 1
        print(f'{base}: {pred_label}  scores={out.tolist()}')

    acc = (correct / max(1, total)) * 100.0
    print(f'Accuracy on {total} files: {acc:.2f}%')


if __name__ == '__main__':
    main()


