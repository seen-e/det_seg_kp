#pragma once

#include <NvInfer.h>

#include <cstdint>
#include <string>
#include <vector>

namespace det_seg_kp {

/** Register ``det_seg_kp::MSDeformAttn`` with the TensorRT plugin registry. */
bool registerMsdaPlugin();

class MSDeformAttnPlugin final : public nvinfer1::IPluginV2DynamicExt {
 public:
  explicit MSDeformAttnPlugin(int im2col_step);
  MSDeformAttnPlugin(const void* data, size_t length);
  MSDeformAttnPlugin(const MSDeformAttnPlugin&) = default;
  ~MSDeformAttnPlugin() override = default;

  const char* getPluginType() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  int getNbOutputs() const noexcept override;
  int initialize() noexcept override;
  void terminate() noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void* buffer) const noexcept override;
  void destroy() noexcept override;
  void setPluginNamespace(const char* pluginNamespace) noexcept override;
  const char* getPluginNamespace() const noexcept override;

  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex,
      const nvinfer1::DimsExprs* inputs,
      int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) noexcept override;
  bool supportsFormatCombination(
      int pos,
      const nvinfer1::PluginTensorDesc* inOut,
      int nbInputs,
      int nbOutputs) noexcept override;
  void configurePlugin(
      const nvinfer1::DynamicPluginTensorDesc* in,
      int nbInputs,
      const nvinfer1::DynamicPluginTensorDesc* out,
      int nbOutputs) noexcept override;
  size_t getWorkspaceSize(
      const nvinfer1::PluginTensorDesc* inputs,
      int nbInputs,
      const nvinfer1::PluginTensorDesc* outputs,
      int nbOutputs) const noexcept override;
  int enqueue(
      const nvinfer1::PluginTensorDesc* inputDesc,
      const nvinfer1::PluginTensorDesc* outputDesc,
      const void* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) noexcept override;
  nvinfer1::DataType getOutputDataType(
      int index,
      const nvinfer1::DataType* inputTypes,
      int nbInputs) const noexcept override;

  int im2col_step() const { return im2col_step_; }

 private:
  int im2col_step_ = 64;
  std::string namespace_;
};

class MSDeformAttnPluginCreator final : public nvinfer1::IPluginCreator {
 public:
  MSDeformAttnPluginCreator();
  const char* getPluginName() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
  nvinfer1::IPluginV2* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
  nvinfer1::IPluginV2* deserializePlugin(
      const char* name, const void* serialData, size_t serialLength) noexcept override;
  void setPluginNamespace(const char* pluginNamespace) noexcept override;
  const char* getPluginNamespace() const noexcept override;

 private:
  std::string namespace_;
  std::vector<nvinfer1::PluginField> fields_;
  nvinfer1::PluginFieldCollection fc_{};
};

}  // namespace det_seg_kp
