#include "det_seg_kp_engine.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(DET_SEG_KP_HAS_OPENCV)
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace {

struct Args {
  std::string engine, image, out = "vis.jpg";
  float score = 0.5f, mask = 0.5f, kp = 0.1f;
};

Args parse(int argc, char** argv) {
  Args a;
  auto need = [&](int& i) -> std::string {
    if (++i >= argc) throw std::runtime_error("missing arg value");
    return argv[i];
  };
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "--engine") a.engine = need(i);
    else if (k == "--image") a.image = need(i);
    else if (k == "--out") a.out = need(i);
    else if (k == "--score") a.score = std::stof(need(i));
    else if (k == "--mask") a.mask = std::stof(need(i));
    else if (k == "--kp") a.kp = std::stof(need(i));
    else throw std::runtime_error("unknown arg: " + k);
  }
  if (a.engine.empty() || a.image.empty()) {
    throw std::runtime_error(
        "usage: demo --engine e.engine --image img.jpg [--score 0.5] [--mask 0.5] [--kp 0.1] [--out vis.jpg]");
  }
  return a;
}

std::vector<float> load_nchw(const std::string& path, int H, int W) {
#if defined(DET_SEG_KP_HAS_OPENCV)
  cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
  if (bgr.empty()) throw std::runtime_error("read failed: " + path);
  cv::Mat rgb;
  cv::resize(bgr, bgr, {W, H});
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
  std::vector<cv::Mat> ch(3);
  cv::split(rgb, ch);
  std::vector<float> out(3 * H * W);
  for (int c = 0; c < 3; ++c)
    std::memcpy(out.data() + c * H * W, ch[c].ptr<float>(), H * W * sizeof(float));
  return out;
#else
  throw std::runtime_error("build with OpenCV to load images");
#endif
}

void save_vis(const std::string& path, const std::string& image,
              const det_seg_kp::InferResult& r, int H, int W) {
#if defined(DET_SEG_KP_HAS_OPENCV)
  cv::Mat bgr = cv::imread(image, cv::IMREAD_COLOR);
  cv::resize(bgr, bgr, {W, H});
  if (r.batches.empty()) {
    cv::imwrite(path, bgr);
    return;
  }
  for (size_t i = 0; i < r.batches[0].size(); ++i) {
    const auto& d = r.batches[0][i];
    cv::Scalar color(40 + 37 * int(i) % 200, 40 + 67 * int(i) % 200, 40 + 97 * int(i) % 200);
    cv::rectangle(bgr,
                  {int(d.box_xyxy[0] * W), int(d.box_xyxy[1] * H)},
                  {int(d.box_xyxy[2] * W), int(d.box_xyxy[3] * H)}, color, 2);
    if (!d.mask.empty()) {
      cv::Mat m(r.mask_h, r.mask_w, CV_8UC1);
      for (int y = 0; y < r.mask_h; ++y)
        for (int x = 0; x < r.mask_w; ++x)
          m.at<uint8_t>(y, x) = d.mask[y * r.mask_w + x] ? 255 : 0;
      cv::Mat big;
      cv::resize(m, big, bgr.size(), 0, 0, cv::INTER_NEAREST);
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
          if (big.at<uint8_t>(y, x)) {
            auto& p = bgr.at<cv::Vec3b>(y, x);
            p[0] = uint8_t(p[0] * 0.5 + color[0] * 0.5);
            p[1] = uint8_t(p[1] * 0.5 + color[1] * 0.5);
            p[2] = uint8_t(p[2] * 0.5 + color[2] * 0.5);
          }
    }
    cv::circle(bgr, {int(d.keypoint_xy[0] * W), int(d.keypoint_xy[1] * H)}, 4, color, -1);
  }
  cv::imwrite(path, bgr);
#else
  (void)path;
  (void)image;
  (void)r;
  (void)H;
  (void)W;
  throw std::runtime_error("build with OpenCV to save vis");
#endif
}

}  // namespace

int main(int argc, char** argv) {
  try {
    auto args = parse(argc, argv);
    det_seg_kp::EngineInfer eng;
    eng.load(args.engine);
    auto img = load_nchw(args.image, eng.input_h(), eng.input_w());
    det_seg_kp::InferConfig cfg{args.score, args.mask, args.kp};
    auto res = eng.infer(img.data(), 1, cfg);
    size_t n = res.batches.empty() ? 0 : res.batches[0].size();
    std::cout << "detections=" << n << "\n";
    for (size_t i = 0; i < n; ++i) {
      const auto& d = res.batches[0][i];
      std::printf("  #%zu score=%.3f box=[%.3f,%.3f,%.3f,%.3f] kp=(%.3f,%.3f)\n", i,
                  d.score, d.box_xyxy[0], d.box_xyxy[1], d.box_xyxy[2], d.box_xyxy[3],
                  d.keypoint_xy[0], d.keypoint_xy[1]);
    }
    save_vis(args.out, args.image, res, eng.input_h(), eng.input_w());
    std::cout << "wrote " << args.out << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
