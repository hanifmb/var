#include "hsolver.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "optimization.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

cv::Point2f pt(-1, -1);
bool new_coords = false;

void mouse_callback(int event, int x, int y, int flag, void *param) {
  if (event == cv::EVENT_LBUTTONDOWN) {
    pt.x = x;
    pt.y = y;
    new_coords = true;
  }
}

void draw_cross(cv::Point pt, cv::Mat image, int size) {

  cv::Point starting1(pt.x - size, pt.y - size);
  cv::Point ending1(pt.x + size, pt.y + size);

  cv::Point starting2(pt.x + size, pt.y - size);
  cv::Point ending2(pt.x - size, pt.y + size);

  cv::Scalar line_Color(255, 0, 0);
  int thickness = 2;

  cv::line(image, starting1, ending1, line_Color, thickness);
  cv::line(image, starting2, ending2, line_Color, thickness);
}

hsolver::ImagePts waitPoints(cv::Mat img) {

  int count = 0;
  char key = 0;
  hsolver::ImagePts image_pts;
  while ((int)key != 27) {

    if (new_coords) {
      draw_cross(pt, img, 15);
      int num_line_pts = 3;
      int num_circle_pts = 7;

      if (count < num_line_pts)
        image_pts.line_pts.emplace_back(cv::Point2f(pt.x, pt.y));
      else
        image_pts.circle_pts.emplace_back(cv::Point2f(pt.x, pt.y));

      new_coords = false;
      if (count == num_line_pts + num_circle_pts - 1)
        break;
      ++count;
    }

    cv::imshow("image", img);
    key = cv::waitKey(1);
  }

  return image_pts;
}

void transformPoint(const cv::Point2f &input, cv::Point2f &output,
                    const cv::Mat &H, bool isPerspective) {
  cv::Mat pt(3, 1, CV_32FC1);
  pt.at<float>(0, 0) = input.x;
  pt.at<float>(1, 0) = input.y;
  pt.at<float>(2, 0) = 1.0;

  cv::Mat ptTransformed = H * pt;
  if (isPerspective)
    ptTransformed = (1.0 / ptTransformed.at<float>(2, 0)) * ptTransformed;

  float newX = (ptTransformed.at<float>(0, 0));
  float newY = (ptTransformed.at<float>(1, 0));

  output = cv::Point2f(newX, newY);
}

void draw_line(cv::Point pt1, cv::Point pt2, cv::Mat image,
               cv::Scalar line_color) {

  int thickness = 2;
  cv::line(image, pt1, pt2, line_color, thickness);
}

int main() {
  const std::string img_file = "image.png";
  const std::string window_name = "image";

  cv::Mat img = cv::imread(img_file, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "Error: Image file is empty." << std::endl;
    return -1;
  }

  cv::namedWindow(window_name, cv::WINDOW_NORMAL);
  cv::setMouseCallback(window_name, mouse_callback);

  // blocking function to wait for user to select points
  hsolver::ImagePts image_pts = waitPoints(img);

  cv::Mat ans = hsolver::calc_h(image_pts);
  cv::Mat inv_final_H = ans.inv();

  // generating unit cirlce for the map
  float r = 9.15;
  std::vector<cv::Point2f> points_map;
  for (int i = 1; i < 360; i++) {
    float angle = static_cast<float>(i) * 3.14159 / 180.0;
    float x = r * cos(angle);
    float y = r * sin(angle);
    points_map.emplace_back(cv::Point2f(x, y));
  }

  // drawing the transformed unit circle as an ellipse
  for (auto &e : points_map) {
    cv::Point2f output;
    transformPoint(e, output, inv_final_H, true);
    draw_cross(output, img, 3);
  }

  // drawing the offside line
  int key = 0;
  while ((int)key != 27) {
    if (new_coords) {
      cv::Point2f line_a;
      cv::Point2f line_b;
      cv::Point2f transformed_pt;

      transformPoint(pt, transformed_pt, ans, true);
      transformPoint(cv::Point2f(transformed_pt.x, 40), line_a, inv_final_H,
                     true);
      transformPoint(cv::Point2f(transformed_pt.x, -40), line_b, inv_final_H,
                     true);

      draw_line(line_a, line_b, img, cv::Scalar(0, 0, 255));
      new_coords = false;
    }
    cv::imshow("image", img);
    key = cv::waitKey(1);
  }

  return 0;
}