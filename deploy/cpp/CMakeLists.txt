cmake_minimum_required(VERSION 3.18)
project(det_seg_kp_trt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

if(DEFINED TensorRT_ROOT)
  list(APPEND CMAKE_PREFIX_PATH "${TensorRT_ROOT}")
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS
    ${TensorRT_ROOT}/include
    /usr/include
    /usr/include/x86_64-linux-gnu
    /usr/local/include
    /opt/tensorrt/include
)
find_library(TENSORRT_NVINFER nvinfer
  HINTS
    ${TensorRT_ROOT}/lib
    ${TensorRT_ROOT}/lib64
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/lib
    /opt/tensorrt/lib
)
find_library(TENSORRT_NVINFER_PLUGIN nvinfer_plugin
  HINTS
    ${TensorRT_ROOT}/lib
    ${TensorRT_ROOT}/lib64
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/lib
    /opt/tensorrt/lib
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER)
  message(FATAL_ERROR
    "TensorRT not found. Pass -DTensorRT_ROOT=/path/to/TensorRT")
endif()

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT nvinfer: ${TENSORRT_NVINFER}")

find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
if(OpenCV_FOUND)
  message(STATUS "OpenCV found: ${OpenCV_VERSION}")
  add_compile_definitions(DET_SEG_KP_HAS_OPENCV=1)
else()
  message(STATUS "OpenCV not found — demo falls back to PPM I/O")
endif()

# ---- MSDeformAttn TensorRT plugin (CUDA-only) ----
add_library(det_seg_kp_msda_plugin SHARED
  plugin/ms_deform_attn_kernel.cu
  plugin/ms_deform_attn_plugin.cpp
)
target_include_directories(det_seg_kp_msda_plugin
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TENSORRT_INCLUDE_DIR}
)
target_link_libraries(det_seg_kp_msda_plugin
  PUBLIC
    ${TENSORRT_NVINFER}
    CUDA::cudart
)
if(TENSORRT_NVINFER_PLUGIN)
  target_link_libraries(det_seg_kp_msda_plugin PUBLIC ${TENSORRT_NVINFER_PLUGIN})
endif()
set_target_properties(det_seg_kp_msda_plugin PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CXX_VISIBILITY_PRESET default
)

# ---- Engine + demo ----
add_library(det_seg_kp_engine STATIC
  src/det_seg_kp_engine.cpp
)
target_include_directories(det_seg_kp_engine
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin
    ${TENSORRT_INCLUDE_DIR}
)
target_link_libraries(det_seg_kp_engine
  PUBLIC
    det_seg_kp_msda_plugin
    ${TENSORRT_NVINFER}
    CUDA::cudart
)

add_executable(det_seg_kp_demo src/demo.cpp)
target_link_libraries(det_seg_kp_demo PRIVATE det_seg_kp_engine)
if(OpenCV_FOUND)
  target_link_libraries(det_seg_kp_demo PRIVATE ${OpenCV_LIBS})
  target_include_directories(det_seg_kp_demo PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

install(TARGETS det_seg_kp_msda_plugin det_seg_kp_engine det_seg_kp_demo
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
install(FILES plugin/ms_deform_attn_plugin.hpp DESTINATION include)
