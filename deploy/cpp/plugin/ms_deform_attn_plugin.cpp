#include "ms_deform_attn_plugin.hpp"

#include "ms_deform_attn_kernel.cuh"

#include <cuda_fp16.h>

#include <cstring>
#include <iostream>
#include <vector>

namespace det_seg_kp {
namespace {

constexpr char kPluginType[] = "MSDeformAttn";
constexpr char kPluginVersion[] = "1";
constexpr char kPluginNamespace[] = "det_seg_kp";

void write_pod(char*& buffer, const void* src, size_t n) {
  std::memcpy(buffer, src, n);
  buffer += n;
}

void read_pod(const char*& buffer, void* dst, size_t n) {
  std::memcpy(dst, buffer, n);
  buffer += n;
}

}  // namespace

MSDeformAttnPlugin::MSDeformAttnPlugin(int im2col_step)
    : im2col_step_(im2col_step), namespace_(kPluginNamespace) {}

MSDeformAttnPlugin::MSDeformAttnPlugin(const void* data, size_t length) {
  const char* buf = static_cast<const char*>(data);
  read_pod(buf, &im2col_step_, sizeof(im2col_step_));
  namespace_ = kPluginNamespace;
  (void)length;
}

const char* MSDeformAttnPlugin::getPluginType() const noexcept { return kPluginType; }
const char* MSDeformAttnPlugin::getPluginVersion() const noexcept {
  return kPluginVersion;
}
int MSDeformAttnPlugin::getNbOutputs() const noexcept { return 1; }
int MSDeformAttnPlugin::initialize() noexcept { return 0; }
void MSDeformAttnPlugin::terminate() noexcept {}
size_t MSDeformAttnPlugin::getSerializationSize() const noexcept {
  return sizeof(im2col_step_);
}
void MSDeformAttnPlugin::serialize(void* buffer) const noexcept {
  char* buf = static_cast<char*>(buffer);
  write_pod(buf, &im2col_step_, sizeof(im2col_step_));
}
void MSDeformAttnPlugin::destroy() noexcept { delete this; }
void MSDeformAttnPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
  namespace_ = pluginNamespace ? pluginNamespace : "";
}
const char* MSDeformAttnPlugin::getPluginNamespace() const noexcept {
  return namespace_.c_str();
}

nvinfer1::IPluginV2DynamicExt* MSDeformAttnPlugin::clone() const noexcept {
  auto* p = new MSDeformAttnPlugin(im2col_step_);
  p->setPluginNamespace(namespace_.c_str());
  return p;
}

nvinfer1::DimsExprs MSDeformAttnPlugin::getOutputDimensions(
    int /*outputIndex*/,
    const nvinfer1::DimsExprs* inputs,
    int /*nbInputs*/,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
  // value: [B, S, M, D], sampling_loc: [B, Q, M, L, P, 2]
  // output: [B, Q, M*D]
  nvinfer1::DimsExprs out;
  out.nbDims = 3;
  out.d[0] = inputs[0].d[0];  // B
  out.d[1] = inputs[3].d[1];  // Q
  out.d[2] = exprBuilder.operation(
      nvinfer1::DimensionOperation::kPROD, *inputs[0].d[2], *inputs[0].d[3]);
  return out;
}

bool MSDeformAttnPlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int nbInputs,
    int /*nbOutputs*/) noexcept {
  // 0 value, 1 spatial_shapes, 2 level_start_index, 3 sampling_loc, 4 attn_weight, 5 output
  const auto& desc = inOut[pos];
  if (pos == 1 || pos == 2) {
    // index tensors: int32 or int64, linear
    const bool ok_type = desc.type == nvinfer1::DataType::kINT32 ||
                         desc.type == nvinfer1::DataType::kINT64;
    return ok_type && desc.format == nvinfer1::TensorFormat::kLINEAR;
  }
  // float / half, same as value (pos 0) for float tensors + output
  if (pos == 0) {
    return (desc.type == nvinfer1::DataType::kFLOAT ||
            desc.type == nvinfer1::DataType::kHALF) &&
           desc.format == nvinfer1::TensorFormat::kLINEAR;
  }
  if (pos == 3 || pos == 4 || pos == nbInputs) {
    return desc.type == inOut[0].type && desc.format == nvinfer1::TensorFormat::kLINEAR;
  }
  return false;
}

void MSDeformAttnPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* /*in*/,
    int /*nbInputs*/,
    const nvinfer1::DynamicPluginTensorDesc* /*out*/,
    int /*nbOutputs*/) noexcept {}

size_t MSDeformAttnPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* /*inputs*/,
    int /*nbInputs*/,
    const nvinfer1::PluginTensorDesc* /*outputs*/,
    int /*nbOutputs*/) const noexcept {
  return 0;
}

nvinfer1::DataType MSDeformAttnPlugin::getOutputDataType(
    int /*index*/,
    const nvinfer1::DataType* inputTypes,
    int /*nbInputs*/) const noexcept {
  return inputTypes[0];
}

int MSDeformAttnPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* /*outputDesc*/,
    const void* const* inputs,
    void* const* outputs,
    void* /*workspace*/,
    cudaStream_t stream) noexcept {
  try {
    const auto& vdesc = inputDesc[0];
    const auto& sdesc = inputDesc[3];
    const int batch = vdesc.dims.d[0];
    const int spatial_size = vdesc.dims.d[1];
    const int num_heads = vdesc.dims.d[2];
    const int channels = vdesc.dims.d[3];
    const int num_query = sdesc.dims.d[1];
    const int num_levels = sdesc.dims.d[3];
    const int num_point = sdesc.dims.d[4];

    // spatial_shapes / level_start_index may be int32 in TRT — promote to int64
    // scratch on device when needed.
    const int64_t* shapes_i64 = nullptr;
    const int64_t* start_i64 = nullptr;
    int64_t* shapes_tmp = nullptr;
    int64_t* start_tmp = nullptr;

    auto free_tmp = [&]() {
      if (shapes_tmp) cudaFree(shapes_tmp);
      if (start_tmp) cudaFree(start_tmp);
    };

    if (inputDesc[1].type == nvinfer1::DataType::kINT64) {
      shapes_i64 = static_cast<const int64_t*>(inputs[1]);
      start_i64 = static_cast<const int64_t*>(inputs[2]);
    } else {
      const int L = inputDesc[1].dims.d[0];
      cudaMalloc(&shapes_tmp, sizeof(int64_t) * L * 2);
      cudaMalloc(&start_tmp, sizeof(int64_t) * L);
      // convert on host via sync copy — L is tiny (<=4)
      std::vector<int32_t> h_shapes(static_cast<size_t>(L) * 2);
      std::vector<int32_t> h_start(static_cast<size_t>(L));
      cudaMemcpyAsync(h_shapes.data(), inputs[1], sizeof(int32_t) * L * 2,
                      cudaMemcpyDeviceToHost, stream);
      cudaMemcpyAsync(h_start.data(), inputs[2], sizeof(int32_t) * L,
                      cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      std::vector<int64_t> h_shapes64(h_shapes.begin(), h_shapes.end());
      std::vector<int64_t> h_start64(h_start.begin(), h_start.end());
      cudaMemcpyAsync(shapes_tmp, h_shapes64.data(), sizeof(int64_t) * L * 2,
                      cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(start_tmp, h_start64.data(), sizeof(int64_t) * L,
                      cudaMemcpyHostToDevice, stream);
      shapes_i64 = shapes_tmp;
      start_i64 = start_tmp;
    }

    if (vdesc.type == nvinfer1::DataType::kFLOAT) {
      msda::ms_deform_attn_cuda_forward<float>(
          stream,
          static_cast<const float*>(inputs[0]),
          shapes_i64,
          start_i64,
          static_cast<const float*>(inputs[3]),
          static_cast<const float*>(inputs[4]),
          static_cast<float*>(outputs[0]),
          batch, spatial_size, num_heads, channels, num_levels, num_query, num_point,
          im2col_step_);
    } else if (vdesc.type == nvinfer1::DataType::kHALF) {
      msda::ms_deform_attn_cuda_forward<__half>(
          stream,
          static_cast<const __half*>(inputs[0]),
          shapes_i64,
          start_i64,
          static_cast<const __half*>(inputs[3]),
          static_cast<const __half*>(inputs[4]),
          static_cast<__half*>(outputs[0]),
          batch, spatial_size, num_heads, channels, num_levels, num_query, num_point,
          im2col_step_);
    } else {
      free_tmp();
      return -1;
    }
    free_tmp();
    return 0;
  } catch (...) {
    return -1;
  }
}

// ---- Creator ----

MSDeformAttnPluginCreator::MSDeformAttnPluginCreator() : namespace_(kPluginNamespace) {
  fields_.emplace_back(
      nvinfer1::PluginField("im2col_step", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  fc_.nbFields = static_cast<int>(fields_.size());
  fc_.fields = fields_.data();
}

const char* MSDeformAttnPluginCreator::getPluginName() const noexcept {
  return kPluginType;
}
const char* MSDeformAttnPluginCreator::getPluginVersion() const noexcept {
  return kPluginVersion;
}
const nvinfer1::PluginFieldCollection* MSDeformAttnPluginCreator::getFieldNames() noexcept {
  return &fc_;
}

nvinfer1::IPluginV2* MSDeformAttnPluginCreator::createPlugin(
    const char* /*name*/, const nvinfer1::PluginFieldCollection* fc) noexcept {
  int im2col_step = 64;
  if (fc) {
    for (int i = 0; i < fc->nbFields; ++i) {
      const auto& f = fc->fields[i];
      if (!f.name) continue;
      if (std::strcmp(f.name, "im2col_step") == 0 && f.data) {
        im2col_step = *static_cast<const int*>(f.data);
      }
    }
  }
  auto* plugin = new MSDeformAttnPlugin(im2col_step);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

nvinfer1::IPluginV2* MSDeformAttnPluginCreator::deserializePlugin(
    const char* /*name*/, const void* serialData, size_t serialLength) noexcept {
  auto* plugin = new MSDeformAttnPlugin(serialData, serialLength);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

void MSDeformAttnPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
  namespace_ = pluginNamespace ? pluginNamespace : "";
}
const char* MSDeformAttnPluginCreator::getPluginNamespace() const noexcept {
  return namespace_.c_str();
}

bool registerMsdaPlugin() {
  static bool registered = false;
  if (registered) return true;
  auto* registry = getPluginRegistry();
  if (!registry) {
    std::cerr << "[msda] getPluginRegistry() returned null\n";
    return false;
  }
  static MSDeformAttnPluginCreator creator;
  registered = registry->registerCreator(creator, kPluginNamespace);
  if (registered) {
    std::cerr << "[msda] registered plugin " << kPluginNamespace << "::" << kPluginType
              << "\n";
  }
  return registered;
}

// Auto-register when the shared library is loaded (trtexec --plugins=...).
static bool g_auto_register = registerMsdaPlugin();

}  // namespace det_seg_kp

// TensorRT also looks for this C symbol in plugin .so files.
extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder* /*finder*/) {}

extern "C" nvinfer1::IPluginCreator* const* getPluginCreators(int32_t* nbCreators) {
  static det_seg_kp::MSDeformAttnPluginCreator creator;
  static nvinfer1::IPluginCreator* creators[] = {&creator};
  *nbCreators = 1;
  return creators;
}
