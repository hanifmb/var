#include "opencv2/core.hpp"
#include "optimization.h"

class hsolver {


public:

    struct Point2f{
        float x, y;
        Point2f(float in_x, float in_y): x(in_x), y(in_y){}
    };

    struct ImagePts{
        std::vector<Point2f> circle_pts;
        std::vector<Point2f> line_pts;
    };

    struct OptimizationInput{
        std::vector<Point2f> circle_img_norm;
        cv::Mat partial_homography; 
    };

    static cv::Mat calc_h(const ImagePts& image_pts);

private:
    hsolver(); 

    static void obj_fun(const alglib::real_1d_array &x, alglib::real_1d_array &fi, void *ptr);
    static std::vector<Point2f> normalize_pts(const std::vector<Point2f> & points);
    static cv::Mat calc_partialh(const std::vector<Point2f>& points_map, const std::vector<Point2f>& points_image);
    static cv::Mat getH(float alpha, float beta, float gamma, float h2, float h3, float h5, float h6, float h8);
};
