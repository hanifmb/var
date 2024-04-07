#include "hsolver.hpp"

std::vector<hsolver::Point2f> hsolver::normalize_pts(const std::vector<hsolver::Point2f> & points){
    // normalize points using synthetic camera
    float H_data[9] = {500, 0, 600, 0, 500, 400, 0, 0, 1};
    cv::Mat H = cv::Mat(3, 3, CV_32F, H_data);
    
    std::vector<hsolver::Point2f> normalized_pts;
    for(auto & e: points){

        cv::Mat pt(3, 1, CV_32FC1);
        pt.at<float>(0, 0) = e.x;
        pt.at<float>(1, 0) = e.y;
        pt.at<float>(2, 0) = 1.0;
        
        cv::Mat ptTransformed = H.inv() * pt;

        float newX = (ptTransformed.at<float>(0, 0));
        float newY = (ptTransformed.at<float>(1, 0));

        normalized_pts.emplace_back(hsolver::Point2f(newX, newY));
    }
    return normalized_pts;
}

void hsolver::obj_fun(const alglib::real_1d_array &x, alglib::real_1d_array &fi, void *ptr)
{
    OptimizationInput* data = static_cast<OptimizationInput*>(ptr);

    float h2=data->partial_homography.at<float>(0, 1);
    float h3=data->partial_homography.at<float>(0, 2);
    float h5=data->partial_homography.at<float>(1, 1);
    float h6=data->partial_homography.at<float>(1, 2);
    float h8=data->partial_homography.at<float>(2, 1);

    for (int i=0; i<7; i++)
    {
        float alpha = x[0];
        float beta = x[1];
        float gamma = x[2];

        float u2 = data->circle_img_norm[i].x;
        float v2 = data->circle_img_norm[i].y;

        float D1 = h5 - h8 * h6;
        float D4 = h8 * h3 - h2;
        float D7 = h2 * h6 - h5 * h3;
        
        float A2 = v2 - h6;
        float A3 = -h8 * v2 + h5;
        float B2 = -u2 + h3;
        float B3 = h8 * u2 - h2;
        float C2 = h6 * u2 - h3 * v2;
        float C3 = -h5 * u2 + h2 * v2;
        
        float D = D1 * u2 + D4 * v2 + D7;
        float R = 9.15;

        float u1 = D / (A3 * alpha + B3 * beta + C3 * gamma);
        float v1 = (A2 * alpha + B2 * beta + C2 * gamma) / (A3 * alpha + B3 * beta + C3 * gamma);
        float d = sqrt(u1 * u1 + v1 * v1) - R;

        fi[i] = d;
    }
}


cv::Mat hsolver::calc_partialh(const std::vector<hsolver::Point2f>& points_map, const std::vector<hsolver::Point2f>& points_image)
{
    const size_t ptsNum = points_image.size();
    cv::Mat A(2 * ptsNum, 6, CV_32F);


    for (int i = 0; i < ptsNum; i++)
    {
        float v1 = points_map[i].y;

        float u2 = points_image[i].x;
        float v2 = points_image[i].y;

        A.at<float>(2 * i, 0) = v1;
        A.at<float>(2 * i, 1) = 1.0f;
        A.at<float>(2 * i, 2) = 0.0f;
        A.at<float>(2 * i, 3) = 0.0f;
        A.at<float>(2 * i, 4) = -u2 * v1;
        A.at<float>(2 * i, 5) = -u2;

        A.at<float>(2 * i + 1, 0) = 0.0f;
        A.at<float>(2 * i + 1, 1) = 0.0f;
        A.at<float>(2 * i + 1, 2) = v1;
        A.at<float>(2 * i + 1, 3) = 1;
        A.at<float>(2 * i + 1, 4) = -v2 * v1;
        A.at<float>(2 * i + 1, 5) = -v2;
    }

    cv::Mat eVecs(6, 6, CV_32F), eVals(6, 6, CV_32F);
    cv::eigen(A.t() * A, eVals, eVecs);

    cv::Mat H = cv::Mat::zeros(3, 3, CV_32F);   

    std::cout << "evecs:\n" << eVecs << "\n\n";
    std::cout << "evals:\n" << eVals << "\n\n";

    int count = 0;
    for (int i = 0; i < 3; i++){
        for (int j = 1; j < 3; j++){
            H.at<float>(i, j) = eVecs.at<float>(5, count);
            count++;
        }
    }

    /* normalize */
    H = H * (1.0 / H.at<float>(2, 2));
    std::cout << "6 H:\n" << H << "\n\n";
    return H;
}


cv::Mat hsolver::getH(float alpha, float beta, float gamma,
        float h2, float h3, float h5, float h6, float h8){

    float D1 = h5 - h8 * h6;
    float D4 = h8 * h3 - h2;
    float D7 = h2 * h6 - h5 * h3;
    
    float a5 = 1;
    float a6 = -h8;
    float a8 = -h6;
    float a9 = h5;
    float b2 = -1;
    float b3 = h8;
    float b8 = h3;
    float b9 = -h2;
    float c2 = h6;
    float c3 = -h5;
    float c5 = -h3;
    float c6 = h2;

    float H_data[9] = { D1, D4, D7, 
                        beta * b2 + gamma * c2, alpha * a5 + gamma * c5, alpha * a8 + beta * b8, 
                        beta * b3 + gamma * c3, alpha * a6 + gamma * c6, alpha * a9 + beta * b9 };
    cv::Mat H = cv::Mat(3, 3, CV_32F, H_data);
    return H.clone();
}

cv::Mat hsolver::calc_h(const ImagePts& image_pts){

    float r = 9.15;
    std::vector<hsolver::Point2f> line_map = {hsolver::Point2f(0, -r), hsolver::Point2f(0, 0), hsolver::Point2f(0, r)};
    std::vector<hsolver::Point2f> line_img_norm = normalize_pts(image_pts.line_pts);

    OptimizationInput op_input;
    op_input.partial_homography = calc_partialh(line_map, line_img_norm);
    op_input.circle_img_norm = normalize_pts(image_pts.circle_pts);

    float h2=op_input.partial_homography.at<float>(0, 1);
    float h3=op_input.partial_homography.at<float>(0, 2);
    float h5=op_input.partial_homography.at<float>(1, 1);
    float h6=op_input.partial_homography.at<float>(1, 2);
    float h8=op_input.partial_homography.at<float>(2, 1);

    // setting up the optimiziation parameters including inital guess x
    alglib::real_1d_array x = "[0.0001,0.0001,0.0001]";
    alglib::real_1d_array s = "[1,1,1]";

    double epsx = 0;
    alglib::ae_int_t maxits = 0;
    alglib::minlmstate state;
    alglib::minlmreport rep;

    alglib::minlmcreatev(3, 7, x, 0.0001, state);
    alglib::minlmsetcond(state, epsx, maxits);
    alglib::minlmsetscale(state, s);

    alglib::minlmoptimize(state, obj_fun, nullptr, &op_input);

    minlmresults(state, x, rep);
    printf("%s\n", x.tostring(8).c_str()); 
    std::cout << "iter "<< rep.iterationscount;

    // calculating the final homography matrix
    float alpha = x[0];
    float beta = x[1];
    float gamma = x[2];

    cv::Mat K = (cv::Mat_<float>(3, 3) << 500, 0, 600, 0, 500, 400, 0, 0, 1);
    cv::Mat final_H = getH(alpha, beta, gamma, h2, h3, h5, h6, h8);
    cv::Mat ans = final_H * K.inv();
    // cv::Mat inv_final_H = ans.inv();

    return ans;
}