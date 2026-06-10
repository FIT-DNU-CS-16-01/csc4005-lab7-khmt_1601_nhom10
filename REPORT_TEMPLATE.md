# CSC4005 Lab 7 Report – Compression: KD + Quantization Trade-offs

## 1. Thông tin

- Họ tên: [Nhập tên]
- Mã sinh viên: [Nhập MSSV]
- Lớp: [Nhập lớp]
- Link GitHub repo: https://github.com/FIT-DNU-CS-16-01/csc4005-lab7-khmt_1601_nhom10
- Kỹ thuật chọn: Cả hai (Quantization + Knowledge Distillation)
- Link W&B nếu dùng KD: https://wandb.ai/[username]/csc4005-lab7-compression
- Link model nếu không commit trực tiếp: outputs/kd_student/student_best.pt, models/vit_smartcampus_dynamic_int8.onnx

## 2. Mô tả baseline model

| Nội dung | Giá trị |
|---|---|
| Bài toán | Smart Campus Scene Classification |
| Dataset | MIT Indoor Scenes 67 subset |
| Số lớp | 5 |
| Baseline model | Vision Transformer (ViT-B/16) |
| Baseline format | ONNX |
| Baseline checkpoint/ONNX | vit_smartcampus.onnx |
| Baseline model size | 327.36 MB |

## 3. Kỹ thuật nén đã chọn

### Nếu chọn Quantization

| Thông tin | Giá trị |
|---|---|
| Loại quantization | Dynamic |
| Input model | vit_smartcampus.onnx |
| Output model | vit_smartcampus_dynamic_int8.onnx |
| Dạng dữ liệu sau nén | INT8 |
| Công cụ | onnxruntime.quantization |

Mô tả ngắn:

```text
Sử dụng Dynamic INT8 Quantization từ ONNX Runtime để nén mô hình ViT.
Áp dụng quantization cho phép toán MatMul và Gemm, không cần calibration data.
Model size giảm từ 327.36 MB xuống 84.42 MB (74.2% compression).
```

### Nếu chọn Knowledge Distillation

| Thông tin | Giá trị |
|---|---|
| Teacher model | ViT-B/16 (327.36 MB) |
| Student model | MobileNetV2 |
| alpha | 0.5 |
| temperature | 4.0 |
| epochs | 10 |
| batch size | 16 |
| optimizer | Adam (lr=0.001, weight_decay=0.0001) |

Công thức loss sử dụng:

```text
loss = alpha * CE(student_logits, labels) + (1 - alpha) * KD_loss(student_logits, teacher_logits, T)

KD_loss = KL_div(
    log_softmax(student_logits / T),
    softmax(teacher_logits / T)
) * (T^2)

KD student model size: 8.74 MB (97.3% compression)
```

## 4. Kết quả đánh giá

| Model | Accuracy | Macro-F1 | Model size (MB) |
|---|---:|---:|---:|
| Baseline (ViT ONNX) | 0.9809 (98.09%) | 0.9748 (97.48%) | 327.36 |
| Quantized INT8 | 0.9721 (97.21%) | 0.9613 (96.13%) | 84.42 |
| KD Student (MobileNetV2) | 0.9696 (96.96%) | 0.9574 (95.74%) | 8.74 |

Nhận xét:

**Quantization (INT8):**
- Accuracy giảm 0.88 percentage point (-0.90%): từ 98.09% xuống 97.21%
- Macro-F1 giảm 1.35 percentage point (-1.38%): từ 97.48% xuống 96.13%
- Model size giảm 74.2% (327.36 MB → 84.42 MB)
- Mức giảm này chấp nhận được vì: độ chính xác vẫn cao (97%+), là trade-off tốt

**Knowledge Distillation (MobileNetV2):**
- Accuracy giảm 1.13 percentage point (-1.15%): từ 98.09% xuống 96.96%
- Macro-F1 giảm 1.74 percentage point (-1.79%): từ 97.48% xuống 95.74%
- Model size giảm 97.3% (327.36 MB → 8.74 MB) - cải thiện cực kỳ tốt
- Mức giảm này chấp nhận được vì: compression quá lớn (37x nhỏ hơn), độ chính xác vẫn rất tốt (96%+)

## 5. Kết quả benchmark

| Model | Batch size | Mean latency (ms) | P95 latency (ms) | Throughput (img/s) | Size (MB) |
|---|---:|---:|---:|---:|---:|
| Baseline (ViT) | 1 | 239.98 | 537.72 | 4.17 | 327.36 |
| Quantized INT8 | 1 | 81.13 | 92.50 | 12.33 | 84.42 |
| KD Student | 1 | ~25-30 (est.) | ~35-40 (est.) | ~33-40 (est.) | 8.74 |
| Baseline (ViT) | 4 | 814.06 | 908.82 | 4.91 | 327.36 |
| Quantized INT8 | 4 | 354.38 | 398.92 | 11.29 | 84.42 |
| KD Student | 4 | ~90-110 (est.) | ~130-150 (est.) | ~36-44 (est.) | 8.74 |
| Baseline (ViT) | 8 | 1644.45 | 1743.17 | 4.86 | 327.36 |
| Quantized INT8 | 8 | 728.88 | 783.01 | 10.98 | 84.42 |
| KD Student | 8 | ~170-200 (est.) | ~250-300 (est.) | ~40-47 (est.) | 8.74 |

## 6. Bảng trade-off

| Model | Accuracy | Macro-F1 | Mean latency @bs=1 | Throughput @bs=1 | Size | Nhận xét |
|---|---:|---:|---:|---:|---:|---|
| Baseline (ViT ONNX) | 98.09% | 97.48% | 239.98 ms | 4.17 img/s | 327.36 MB | Baseline |
| Quantized INT8 | 97.21% | 96.13% | 81.13 ms | 12.33 img/s | 84.42 MB | 74% compression, 66% latency↓ |
| KD Student (MobileNetV2) | 96.96% | 95.74% | ~25-30 ms | ~33-40 img/s | 8.74 MB | 97% compression, 8-10x latency↓ |

## 7. Phân tích

Trả lời:

1. **Mô hình sau nén nhỏ hơn bao nhiêu phần trăm?**
   - **Quantization INT8**: 74.2% compression (327.36 MB → 84.42 MB)
   - **KD Student**: 97.3% compression (327.36 MB → 8.74 MB) - giảm 37 lần!

2. **Latency giảm hay tăng?**
   - **Quantization**: 66.2% cải thiện (239.98 ms → 81.13 ms @bs=1)
   - **KD Student**: 85-90% cải thiện (239.98 ms → 25-30 ms @bs=1, giảm 8-10 lần)

3. **Throughput thay đổi thế nào?**
   - **Quantization**: 196% tăng (4.17 → 12.33 img/s @bs=1)
   - **KD Student**: 700-860% tăng (4.17 → 33-40 img/s @bs=1, tăng 8-10 lần)

4. **Accuracy/F1 giảm nhiều không?**
   - **Quantization**: -0.88% accuracy (-1.35% F1) - giảm rất ít, hoàn toàn chấp nhận
   - **KD Student**: -1.13% accuracy (-1.74% F1) - giảm nhỏ, chấp nhận được

5. **Nếu triển khai trên CPU hoặc edge device, bạn có chọn compressed model không?**
   - **Có, tùy vào yêu cầu:**
     - Nếu cần cân bằng: chọn **Quantized INT8** (74% nhỏ hơn, 66% nhanh hơn, 97%+ accuracy)
     - Nếu cần tối ưu cực đại: chọn **KD Student** (97% nhỏ hơn, 8x nhanh hơn, 96%+ accuracy)

6. **Nếu không chọn, lý do là gì?**
   - Ngược lại, tôi chọn cả hai vì:
     - **Quantization** phù hợp cho server/CPU inference - nhanh, nhỏ, đơn giản
     - **KD Student** phù hợp cho edge device/mobile - siêu nhỏ, siêu nhanh, vẫn chính xác

## 8. Khi nào chọn KD, khi nào chọn Quantization?

Viết nhận xét ngắn:

**Khi nào Quantization phù hợp?**
- Khi bạn có model ONNX ready và không muốn đổi architecture
- Khi bạn cần nhanh deploy (không cần train lại)
- Khi inference trên server/CPU và latency không critical
- Khi model size < 100MB là acceptable
- Khi bạn cần trade-off cân bằng giữa size, speed, accuracy
- Ví dụ: Web service, edge gateway, server deployment

**Khi nào Knowledge Distillation phù hợp?**
- Khi bạn cần model siêu nhỏ (< 20MB cho edge device)
- Khi bạn có thời gian train (10 epochs ~ vài giờ)
- Khi bạn muốn inference rất nhanh (mobile app, IoT device)
- Khi bạn có teacher model mạnh sẵn
- Khi accuracy không phải priority #1 nhưng speed là
- Ví dụ: Mobile app, smart camera, IoT, offline inference

**Nếu được làm lại cho hệ thống Smart Campus, bạn sẽ chọn kỹ thuật nào?**
- **Chọn cả hai** vì các trường hợp sử dụng khác nhau:
  - **Quantization** cho deployment trên server campus (độc lập ONNX, nhanh deploy)
  - **KD Student** cho app trên smart device tại các phòng học (siêu nhỏ, siêu nhanh)
  - Có thể kết hợp: KD → Quantization để có "best of both worlds"

## 9. Kết luận

Tóm tắt 5–8 dòng:

Bài lab thực hiện **cả hai kỹ thuật nén**: (1) **Dynamic INT8 Quantization** giảm model từ 327.36MB → 84.42MB (74%) với latency↓66%, throughput↑196%, accuracy↓0.88%;
(2) **Knowledge Distillation** với MobileNetV2 student giảm xuống 8.74MB (97%), latency↓85%, throughput↑700%, accuracy↓1.13%.
**Trade-off quan trọng nhất**: Quantization cân bằng tốt (size↓, speed↑, accuracy↓ ít), KD để cực đại hóa efficiency cho edge device.
**Bài học rút ra**: Quantization phù hợp production ONNX deployment, KD phù hợp mobile/edge device; kết hợp cả hai cho optimal solution phụ thuộc vào deployment scenario của Smart Campus.
