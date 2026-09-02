#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace det_seg_kp {

struct InferConfig {
  float score_threshold = 0.5f;
  float mask_threshold = 0.5f;
  float kp_threshold = 0.1f;
};

struct Detection {
  float score = 0.f;
  int label = 0;
  float box_xyxy[4]{};   // normalized
  float keypoint_xy[2]{};
  std::vector<uint8_t> mask;  // Hm*Wm
};

struct InferResult {
  int batch = 0, num_queries = 0, mask_h = 0, mask_w = 0;
  std::vector<std::vector<Detection>> batches;
};

/** TensorRT runner. I/O names match build_onnx.py; registers MSDA plugin on load. */
class EngineInfer {
 public:
  EngineInfer();
  ~EngineInfer();
  EngineInfer(const EngineInfer&) = delete;
  EngineInfer& operator=(const EngineInfer&) = delete;

  void load(const std::string& engine_path);
  InferResult infer(const float* images_nchw, int batch, const InferConfig& cfg);

  int input_h() const { return input_h_; }
  int input_w() const { return input_w_; }
  int num_queries() const { return num_queries_; }
  int mask_h() const { return mask_h_; }
  int mask_w() const { return mask_w_; }
  int max_batch() const { return max_batch_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  int input_h_ = 0, input_w_ = 0, num_queries_ = 0;
  int mask_h_ = 0, mask_w_ = 0, max_batch_ = 1;
};

}  // namespace det_seg_kp
