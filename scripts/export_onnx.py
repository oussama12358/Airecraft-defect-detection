import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch, argparse
from omegaconf import OmegaConf
from src.models.efficientnet_b3 import build_efficientnet_b3
from src.models.resnet50     import build_resnet50
from src.models.baseline_cnn import BaselineCNN
from src.training.lora import apply_lora


def export(checkpoint: str, output: str, model_name: str, num_classes: int):
    print(f"[ONNX] Loading {model_name}...")

    if model_name == "efficientnet_b3":
        model = build_efficientnet_b3(num_classes)
    elif model_name == "resnet50":
        model = build_resnet50(num_classes)
    else:
        model = BaselineCNN(num_classes)

    cfg = OmegaConf.to_container(OmegaConf.load("configs/config.yaml"), resolve=True)
    if cfg["training"].get("use_lora", False):
        print("[ONNX] Applying LoRA wrappers before loading checkpoint...")
        model = apply_lora(
            model,
            r=cfg["training"]["lora_rank"],
            alpha=cfg["training"]["lora_alpha"],
            dropout=cfg["training"]["lora_dropout"],
            target_modules=cfg["training"].get("lora_target_modules", ["fc", "classifier"]),
        )

    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 224, 224)

    try:
        torch.onnx.export(
            model, dummy, output,
            export_params=True, opset_version=17,
            do_constant_folding=True,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
    except ModuleNotFoundError as e:
        if "onnxscript" in str(e):
            raise RuntimeError(
                "onnx export requires onnxscript. Install it with `pip install onnxscript` "
                "or run `pip install -r requirements.txt`."
            ) from e
        raise

    import onnx
    onnx.checker.check_model(onnx.load(output))

    pt_mb   = Path(checkpoint).stat().st_size / 1e6
    onnx_mb = Path(output).stat().st_size / 1e6
    print(f"[ONNX] Saved → {output}")
    print(f"[ONNX] PyTorch: {pt_mb:.1f} MB  →  ONNX: {onnx_mb:.1f} MB")


def benchmark(onnx_path: str, runs: int = 100):
    import onnxruntime as ort, numpy as np, time

    sess  = ort.InferenceSession(onnx_path)
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    name  = sess.get_inputs()[0].name

    for _ in range(10): sess.run(None, {name: dummy})

    t = time.perf_counter()
    for _ in range(runs): sess.run(None, {name: dummy})
    ms = (time.perf_counter() - t) / runs * 1000
    print(f"[ONNX] Avg latency ({runs} runs): {ms:.2f} ms")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--output",       default="checkpoints/model.onnx")
    p.add_argument("--model",        default="efficientnet_b3",
                   choices=["efficientnet_b3", "resnet50", "baseline_cnn"])
    p.add_argument("--num_classes",  type=int, default=6)
    p.add_argument("--benchmark",    action="store_true")
    args = p.parse_args()

    export(args.checkpoint, args.output, args.model, args.num_classes)
    if args.benchmark:
        benchmark(args.output)