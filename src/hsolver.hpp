#include "opencv2/core.hpp"
#include "optimization.h"

class hsolver {

public:
  struct ImagePts {
    std::vector<cv::Point2f> circle_pts;
    std::vector<cv::Point2f> line_pts;
  };

  struct OptimizationInput {
    std::vector<cv::Point2f> circle_img_norm;
    std::vector<float> partialh;
  };

  static cv::Mat calc_h(const ImagePts &image_pts);

private:
  hsolver();

  static void obj_fun(const alglib::real_1d_array &x, alglib::real_1d_array &fi,
                      void *ptr);

  static std::vector<cv::Point2f>
  normalize_pts(const std::vector<cv::Point2f> &points);
  static std::vector<float>
  calc_partialh(const std::vector<cv::Point2f> &points_map,
                const std::vector<cv::Point2f> &points_image);

  static cv::Mat construct_h(const float &alpha, const float &beta,
                             const float &gamma,
                             const std::vector<float> &partialh);
};
