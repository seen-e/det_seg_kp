# Deploy (CUDA-only)

```text
deploy/
  model.py           # DetSegKPDeploy + MSDA patch + integral KP + load
  build_onnx.py
  build_engine.py
  verify.py
  cpp/               # TRT engine + MSDeformAttn plugin
```

```bash
# build plugin / demo
cmake -S deploy/cpp -B deploy/cpp/build -DTensorRT_ROOT=...
cmake --build deploy/cpp/build -j

# smoke / export / engine
CUDA_VISIBLE_DEVICES=0 python deploy/verify.py --device cuda:0
CUDA_VISIBLE_DEVICES=0 python deploy/build_onnx.py --checkpoint ckpt.pth --device cuda:0 --verify
python deploy/build_engine.py --onnx deploy/artifacts/det_seg_kp.onnx --fp16

./deploy/cpp/build/det_seg_kp_demo --engine deploy/artifacts/det_seg_kp_fp16.engine --image x.jpg
```

ONNX node: `det_seg_kp::MSDeformAttn` ← `libdet_seg_kp_msda_plugin.so`  
Outputs: scores / labels / boxes(xyxy) / masks / keypoints（图内阈值 + 积分）
