"""
Export EfficientCrackNet to ONNX — run this on your PC (not Jetson).

Usage:
    python3 jetson/export_onnx.py
    python3 jetson/export_onnx.py --model_path results/saved_models/EfficientCrackNet/best_model_num_real_4.pt
    python3 jetson/export_onnx.py --output results/efficientcracknet.onnx

Then copy the .onnx to Jetson and run jetson/build_trt.sh to compile the engine.
"""

import argparse
import torch
from crack_detection.models.efficientcracknet import EfficientCrackNet


def export(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = EfficientCrackNet().to(device)
    model.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=False)["model_state_dict"]
    )
    model.eval()
    print("Model loaded.")

    dummy = torch.zeros(1, 3, 512, 512, device=device)

    # Disable Flash/efficient attention — forces decomposition into matmul+softmax
    # which TensorRT 10 can compile, unlike the fused sdp attention patterns.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    print(f"Exporting to {args.output} ...")
    torch.onnx.export(
        model,
        dummy,
        args.output,
        opset_version=17,
        input_names=["image"],
        output_names=["mask"],
        do_constant_folding=True,
        dynamo=False,   # use legacy TorchScript exporter — avoids aten::unsafe_view in output
    )
    print("Export done.")

    # Quick verification
    import onnx
    m = onnx.load(args.output)
    onnx.checker.check_model(m)
    print("ONNX model verified OK.")
    print(f"\nNext step — copy to Jetson and run:")
    print(f"  scp {args.output} jetson:~/soil-crack-detection/{args.output}")
    print(f"  ssh jetson 'cd ~/soil-crack-detection && bash jetson/build_trt.sh'")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="results/saved_models/EfficientCrackNet/best_model_num_real_6.pt")
    p.add_argument("--output", default="results/efficientcracknet.onnx")
    return p.parse_args()


if __name__ == "__main__":
    export(parse_args())
