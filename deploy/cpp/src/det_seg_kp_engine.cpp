#include "det_seg_kp_engine.hpp"
#include "ms_deform_attn_plugin.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace det_seg_kp {
namespace {

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TRT] " << msg << std::endl;
    }
  }
};

TrtLogger g_logger;

void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

std::vector<char> read_file(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("failed to open engine: " + path);
  }
  ifs.seekg(0, std::ios::end);
  const auto size = static_cast<size_t>(ifs.tellg());
  ifs.seekg(0, std::ios::beg);
  std::vector<char> buf(size);
  ifs.read(buf.data(), static_cast<std::streamsize>(size));
  return buf;
}

int64_t volume(const nvinfer1::Dims& d) {
  int64_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) {
    const int64_t dim = d.d[i];
    v *= dim > 0 ? dim : 1;
  }
  return v;
}

size_t dtype_size(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kINT64:
      return 8;
    case nvinfer1::DataType::kINT8:
      return 1;
    case nvinfer1::DataType::kBOOL:
      return 1;
    default:
      return 4;
  }
}

bool is_input_tensor(nvinfer1::ICudaEngine* engine, const char* name) {
  return engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
}

}  // namespace

struct EngineInfer::Impl {
  std::unique_ptr<nvinfer1::IRuntime, void (*)(nvinfer1::IRuntime*)> runtime{
      nullptr, [](nvinfer1::IRuntime* p) { delete p; }};
  std::unique_ptr<nvinfer1::ICudaEngine, void (*)(nvinfer1::ICudaEngine*)> engine{
      nullptr, [](nvinfer1::ICudaEngine* p) { delete p; }};
  std::unique_ptr<nvinfer1::IExecutionContext, void (*)(nvinfer1::IExecutionContext*)>
      context{nullptr, [](nvinfer1::IExecutionContext* p) { delete p; }};

  cudaStream_t stream = nullptr;

  const std::string images_name = "images";
  const std::string score_thr_name = "score_threshold";
  const std::string mask_thr_name = "mask_threshold";
  const std::string kp_thr_name = "kp_threshold";
  const std::string scores_name = "scores";
  const std::string labels_name = "labels";
  const std::string boxes_name = "boxes";
  const std::string masks_name = "masks";
  const std::string kps_name = "keypoints";

  std::unordered_map<std::string, void*> device_buffers;
  std::unordered_map<std::string, size_t> buffer_bytes;

  std::vector<float> host_scores;
  std::vector<int64_t> host_labels_i64;
  std::vector<int32_t> host_labels_i32;
  std::vector<float> host_boxes;
  std::vector<float> host_masks;
  std::vector<float> host_kps;
  bool labels_is_i64 = true;

  ~Impl() {
    for (auto& kv : device_buffers) {
      if (kv.second) {
        cudaFree(kv.second);
      }
    }
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }

  void* ensure_buffer(const char* name, size_t bytes) {
    auto it = device_buffers.find(name);
    if (it == device_buffers.end() || buffer_bytes[name] < bytes) {
      if (it != device_buffers.end() && it->second) {
        cudaFree(it->second);
      }
      void* ptr = nullptr;
      check_cuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
      device_buffers[name] = ptr;
      buffer_bytes[name] = bytes;
    }
    return device_buffers[name];
  }

  void bind_all() {
    const int n = engine->getNbIOTensors();
    for (int i = 0; i < n; ++i) {
      const char* name = engine->getIOTensorName(i);
      const nvinfer1::Dims dims = context->getTensorShape(name);
      const auto dt = engine->getTensorDataType(name);
      const size_t bytes = static_cast<size_t>(volume(dims)) * dtype_size(dt);
      void* ptr = ensure_buffer(name, bytes);
      if (!context->setTensorAddress(name, ptr)) {
        throw std::runtime_error(std::string("setTensorAddress failed: ") + name);
      }
    }
  }
};

EngineInfer::EngineInfer() : impl_(std::make_unique<Impl>()) {}
EngineInfer::~EngineInfer() = default;

void EngineInfer::load(const std::string& engine_path) {
  if (!registerMsdaPlugin()) {
    throw std::runtime_error("failed to register det_seg_kp::MSDeformAttn plugin");
  }
  auto blob = read_file(engine_path);
  impl_->runtime.reset(nvinfer1::createInferRuntime(g_logger));
  if (!impl_->runtime) {
    throw std::runtime_error("createInferRuntime failed");
  }
  impl_->engine.reset(impl_->runtime->deserializeCudaEngine(blob.data(), blob.size()));
  if (!impl_->engine) {
    throw std::runtime_error("deserializeCudaEngine failed");
  }
  impl_->context.reset(impl_->engine->createExecutionContext());
  if (!impl_->context) {
    throw std::runtime_error("createExecutionContext failed");
  }
  check_cuda(cudaStreamCreate(&impl_->stream), "cudaStreamCreate");

  const nvinfer1::Dims idims = impl_->engine->getTensorShape(impl_->images_name.c_str());
  if (idims.nbDims != 4) {
    throw std::runtime_error("images tensor must be 4D NCHW");
  }
  max_batch_ = idims.d[0] > 0 ? idims.d[0] : 1;
  input_h_ = idims.d[2] > 0 ? idims.d[2] : 0;
  input_w_ = idims.d[3] > 0 ? idims.d[3] : 0;

  const nvinfer1::Dims sdims = impl_->engine->getTensorShape(impl_->scores_name.c_str());
  num_queries_ = (sdims.nbDims >= 2 && sdims.d[sdims.nbDims - 1] > 0)
                     ? sdims.d[sdims.nbDims - 1]
                     : 0;

  const nvinfer1::Dims mdims = impl_->engine->getTensorShape(impl_->masks_name.c_str());
  if (mdims.nbDims == 4) {
    mask_h_ = mdims.d[2] > 0 ? mdims.d[2] : 0;
    mask_w_ = mdims.d[3] > 0 ? mdims.d[3] : 0;
  }

  const auto labels_dt = impl_->engine->getTensorDataType(impl_->labels_name.c_str());
  impl_->labels_is_i64 = (labels_dt == nvinfer1::DataType::kINT64);
}

InferResult EngineInfer::infer(const float* images, int batch, const InferConfig& cfg) {
  if (!impl_->engine || !impl_->context) {
    throw std::runtime_error("engine not loaded");
  }
  if (batch <= 0 || batch > max_batch_) {
    throw std::runtime_error("invalid batch size");
  }

  // images: [B,3,H,W]
  {
    nvinfer1::Dims idims = impl_->engine->getTensorShape(impl_->images_name.c_str());
    idims.d[0] = batch;
    if (idims.d[2] <= 0) idims.d[2] = input_h_;
    if (idims.d[3] <= 0) idims.d[3] = input_w_;
    if (!impl_->context->setInputShape(impl_->images_name.c_str(), idims)) {
      throw std::runtime_error("setInputShape(images) failed");
    }
  }
  // scalar thresholds: [1]
  for (const auto& name :
       {impl_->score_thr_name, impl_->mask_thr_name, impl_->kp_thr_name}) {
    nvinfer1::Dims tdims;
    tdims.nbDims = 1;
    tdims.d[0] = 1;
    if (is_input_tensor(impl_->engine.get(), name.c_str())) {
      impl_->context->setInputShape(name.c_str(), tdims);
    }
  }

  // Refresh derived dims after shape set.
  {
    const nvinfer1::Dims sdims =
        impl_->context->getTensorShape(impl_->scores_name.c_str());
    if (sdims.nbDims >= 2) {
      num_queries_ = sdims.d[1];
    }
    const nvinfer1::Dims mdims =
        impl_->context->getTensorShape(impl_->masks_name.c_str());
    if (mdims.nbDims == 4) {
      mask_h_ = mdims.d[2];
      mask_w_ = mdims.d[3];
    }
    const nvinfer1::Dims idims =
        impl_->context->getTensorShape(impl_->images_name.c_str());
    input_h_ = idims.d[2];
    input_w_ = idims.d[3];
  }

  impl_->bind_all();

  const size_t img_elems = static_cast<size_t>(batch) * 3u * input_h_ * input_w_;
  check_cuda(
      cudaMemcpyAsync(
          impl_->device_buffers[impl_->images_name],
          images,
          img_elems * sizeof(float),
          cudaMemcpyHostToDevice,
          impl_->stream),
      "H2D images");

  const float thr_score = cfg.score_threshold;
  const float thr_mask = cfg.mask_threshold;
  const float thr_kp = cfg.kp_threshold;
  check_cuda(
      cudaMemcpyAsync(
          impl_->device_buffers[impl_->score_thr_name],
          &thr_score,
          sizeof(float),
          cudaMemcpyHostToDevice,
          impl_->stream),
      "H2D score_thr");
  check_cuda(
      cudaMemcpyAsync(
          impl_->device_buffers[impl_->mask_thr_name],
          &thr_mask,
          sizeof(float),
          cudaMemcpyHostToDevice,
          impl_->stream),
      "H2D mask_thr");
  check_cuda(
      cudaMemcpyAsync(
          impl_->device_buffers[impl_->kp_thr_name],
          &thr_kp,
          sizeof(float),
          cudaMemcpyHostToDevice,
          impl_->stream),
      "H2D kp_thr");

  if (!impl_->context->enqueueV3(impl_->stream)) {
    throw std::runtime_error("enqueueV3 failed");
  }

  const int Q = num_queries_;
  const int Hm = mask_h_;
  const int Wm = mask_w_;
  const size_t n_scores = static_cast<size_t>(batch) * static_cast<size_t>(Q);
  const size_t n_boxes = n_scores * 4u;
  const size_t n_masks = n_scores * static_cast<size_t>(Hm) * static_cast<size_t>(Wm);
  const size_t n_kps = n_scores * 2u;

  impl_->host_scores.resize(n_scores);
  impl_->host_boxes.resize(n_boxes);
  impl_->host_masks.resize(n_masks);
  impl_->host_kps.resize(n_kps);
  if (impl_->labels_is_i64) {
    impl_->host_labels_i64.resize(n_scores);
  } else {
    impl_->host_labels_i32.resize(n_scores);
  }

  auto d2h = [&](const std::string& name, void* host, size_t bytes) {
    check_cuda(
        cudaMemcpyAsync(
            host,
            impl_->device_buffers[name],
            bytes,
            cudaMemcpyDeviceToHost,
            impl_->stream),
        ("D2H " + name).c_str());
  };

  d2h(impl_->scores_name, impl_->host_scores.data(), n_scores * sizeof(float));
  if (impl_->labels_is_i64) {
    d2h(impl_->labels_name, impl_->host_labels_i64.data(), n_scores * sizeof(int64_t));
  } else {
    d2h(impl_->labels_name, impl_->host_labels_i32.data(), n_scores * sizeof(int32_t));
  }
  d2h(impl_->boxes_name, impl_->host_boxes.data(), n_boxes * sizeof(float));
  d2h(impl_->masks_name, impl_->host_masks.data(), n_masks * sizeof(float));
  d2h(impl_->kps_name, impl_->host_kps.data(), n_kps * sizeof(float));
  check_cuda(cudaStreamSynchronize(impl_->stream), "stream sync");

  InferResult result;
  result.batch = batch;
  result.num_queries = Q;
  result.mask_h = Hm;
  result.mask_w = Wm;
  result.batches.resize(static_cast<size_t>(batch));

  for (int b = 0; b < batch; ++b) {
    auto& dets = result.batches[static_cast<size_t>(b)];
    for (int q = 0; q < Q; ++q) {
      const size_t idx = static_cast<size_t>(b) * static_cast<size_t>(Q) + static_cast<size_t>(q);
      const float score = impl_->host_scores[idx];
      // Engine already applied score_threshold; invalid queries are zeroed.
      if (score <= 0.f) {
        continue;
      }
      Detection det;
      det.score = score;
      det.label = impl_->labels_is_i64
                      ? static_cast<int>(impl_->host_labels_i64[idx])
                      : static_cast<int>(impl_->host_labels_i32[idx]);
      std::memcpy(det.box_xyxy, &impl_->host_boxes[idx * 4], 4 * sizeof(float));
      std::memcpy(det.keypoint_xy, &impl_->host_kps[idx * 2], 2 * sizeof(float));
      det.mask.resize(static_cast<size_t>(Hm) * static_cast<size_t>(Wm));
      const float* src = &impl_->host_masks[idx * static_cast<size_t>(Hm) * static_cast<size_t>(Wm)];
      for (size_t i = 0; i < det.mask.size(); ++i) {
        det.mask[i] = src[i] > 0.5f ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
      }
      dets.push_back(std::move(det));
    }
  }
  return result;
}

}  // namespace det_seg_kp
