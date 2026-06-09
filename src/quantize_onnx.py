from __future__ import annotations

import argparse
from pathlib import Path
import onnx  # Dùng để dọn sạch shape lỗi hệ thống

from onnxruntime.quantization import QuantType, quantize_dynamic
from src.utils import file_size_mb, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Apply dynamic quantization to ONNX model.")
    parser.add_argument("--input_onnx", type=str, required=True)
    parser.add_argument("--output_onnx", type=str, required=True)
    parser.add_argument("--mode", type=str, default="dynamic", choices=["dynamic"])
    parser.add_argument("--weight_type", type=str, default="QInt8", choices=["QInt8", "QUInt8"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_onnx)
    output_path = Path(args.output_onnx)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input ONNX not found: {input_path}")

    model = onnx.load(str(input_path))
    
    # 1. Xóa shape ghim ở các đường truyền nội bộ (Nơi chứa lỗi 768 vs 5)
    model.graph.ClearField("value_info")
    
    # 2. Xóa shape ghim ở cổng đầu ra cuối cùng
    for output in model.graph.output:
        if output.type.HasField("tensor_type"):
            output.type.tensor_type.ClearField("shape")
            
    onnx.save(model, str(input_path))
    # ==============================================================================

    weight_type = QuantType.QInt8 if args.weight_type == "QInt8" else QuantType.QUInt8

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=weight_type,
    )

    report = {
        "method": "onnx_dynamic_quantization",
        "input_onnx": str(input_path),
        "output_onnx": str(output_path),
        "weight_type": args.weight_type,
        "baseline_size_mb": file_size_mb(input_path),
        "compressed_size_mb": file_size_mb(output_path),
        "size_reduction_percent": (1 - file_size_mb(output_path) / file_size_mb(input_path)) * 100,
    }
    save_json(report, output_path.parent / "quantization_report.json")
    print(report)


if __name__ == "__main__":
    main()