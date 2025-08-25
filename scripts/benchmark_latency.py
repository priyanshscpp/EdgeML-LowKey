import argparse
import time
import numpy as np
import tensorflow as tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--runs', type=int, default=200)
    args = ap.parse_args()

    interpreter = tf.lite.Interpreter(model_path=args.model, num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    x = np.zeros(input_details['shape'], dtype=np.int8)

    for _ in range(args.warmup):
        interpreter.set_tensor(input_details['index'], x)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details['index'])

    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter_ns()
        interpreter.set_tensor(input_details['index'], x)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details['index'])
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)

    arr = np.array(times)
    print(f'Latency ms: avg {arr.mean():.3f}, p50 {np.percentile(arr,50):.3f}, p90 {np.percentile(arr,90):.3f}, p99 {np.percentile(arr,99):.3f}, min {arr.min():.3f}, max {arr.max():.3f}')


if __name__ == '__main__':
    main()


