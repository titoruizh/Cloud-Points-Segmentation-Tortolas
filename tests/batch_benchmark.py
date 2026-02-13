#!/usr/bin/env python3
"""
Benchmark de batch sizes para la app de inferencia.

Uso:
    python tests/batch_benchmark.py --input data/raw/DEM_MP_250310_CLASIFICADO.laz \
        --checkpoint checkpoints/SWEEP_RTX\ 5090\ PointNet2\ V6\ \(0.25m\)/LR0.0010_W20_J0.005_R3.5_BEST_IOU.pth \
        --batches 64 96 128 160 --num_points 2048 --no-compile

El script carga el modelo (una vez), y para cada batch_size ejecuta
`InferenceEngine.run_inference` midiendo tiempo y pico de memoria GPU
consultando `nvidia-smi` en un hilo paralelo.
"""
import argparse
import os
import time
import threading
import subprocess
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, PROJECT_ROOT)

from app_inference.core.inference_engine import InferenceEngine, InferenceConfig


def gpu_monitor(stop_event, interval=0.2):
    """Polls nvidia-smi and returns peak memory used (MB)."""
    peak = 0
    while not stop_event.is_set():
        try:
            out = subprocess.check_output([
                "nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"
            ])
            vals = out.decode().strip().splitlines()
            if vals:
                used = max(int(v.strip()) for v in vals)
                if used > peak:
                    peak = used
        except Exception:
            pass
        time.sleep(interval)
    return peak


def monitor_thread_func(stop_event, result_holder, interval=0.2):
    result_holder['peak'] = gpu_monitor(stop_event, interval)


def run_bench(input_file, checkpoint, batches, num_points=2048, use_compile=True):
    results = []
    cfg = InferenceConfig(batch_size=int(batches[0]), num_points=num_points, use_compile=use_compile)
    engine = InferenceEngine(cfg)

    print("[+] Cargando modelo (esto puede tardar)...")
    ok = engine.load_model(checkpoint, lambda msg: print("    ", msg))
    if not ok:
        print("Error al cargar el checkpoint, abortando.")
        return results

    for b in batches:
        print(f"\n[+] Ejecutando batch_size={b}")
        engine.config.batch_size = int(b)

        stop_event = threading.Event()
        monitor_res = {}
        t = threading.Thread(target=monitor_thread_func, args=(stop_event, monitor_res))
        t.start()

        start = time.time()
        res = engine.run_inference(input_file, os.path.join("data/predictions/bench", f"bench_bs{b}.laz"), lambda m: print("   ", m), confidence=0.5)
        elapsed = time.time() - start

        stop_event.set()
        t.join()

        peak_mem = monitor_res.get('peak', 0)
        print(f"[+] Resultado: success={res.success} time={elapsed:.1f}s peak_mem={peak_mem}MB")
        results.append({"batch": b, "success": bool(res.success), "time_s": elapsed, "peak_mem_mb": peak_mem, "machinery_points": res.machinery_points, "ground_points": res.ground_points})

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[64, 96, 128])
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--no-compile', dest='use_compile', action='store_false')
    parser.add_argument('--out', default='tests/bench_results.json')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs('data/predictions/bench', exist_ok=True)

    print(f"Proyecto: {PROJECT_ROOT}")
    print(f"Input: {args.input}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Batches: {args.batches}")

    results = run_bench(args.input, args.checkpoint, args.batches, num_points=args.num_points, use_compile=args.use_compile)

    with open(args.out, 'w') as f:
        json.dump({"input": args.input, "checkpoint": args.checkpoint, "results": results}, f, indent=2)

    print(f"Resultados guardados en {args.out}")


if __name__ == '__main__':
    main()
