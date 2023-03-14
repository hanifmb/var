#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "optimization.h"

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"

using namespace alglib;

float h2, h3, h5, h6, h8;
std::vector<cv::Point2f> circle_img_norm;

cv::Point2f pt(-1, -1);
bool new_coords = false;

cv::Mat calcHomography(const std::vector<cv::Point2f>& points_map, const std::vector<cv::Point2f>& points_image)
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
    /* cv::Mat H(3, 2, CV_32F); */   

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
    /* return normalized_points_map.T.inv() * H * normalized_points_image.T; */
}

void normalizeWithSyntheticCam(const std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& normalized_points){
    
    float H_data[9] = {500, 0, 600, 0, 500, 400, 0, 0, 1};
    cv::Mat H = cv::Mat(3, 3, CV_32F, H_data);
    
    for(auto & e: points){

        cv::Mat pt(3, 1, CV_32FC1);
        pt.at<float>(0, 0) = e.x;
        pt.at<float>(1, 0) = e.y;
        pt.at<float>(2, 0) = 1.0;
        
        cv::Mat ptTransformed = H.inv() * pt;

        /* ptTransformed = (1.0 / ptTransformed.at<float>(2, 0)) * ptTransformed; */

        float newX = (ptTransformed.at<float>(0, 0));
        float newY = (ptTransformed.at<float>(1, 0));

        normalized_points.emplace_back(cv::Point2f(newX, newY));
    }
}

void  function1_fvec(const real_1d_array &x, real_1d_array &fi, void *ptr)
{


    for (int i=0; i<7; i++)
    {

        float alpha = x[0];
        float beta = x[1];
        float gamma = x[2];

        float u2 = circle_img_norm[i].x;
        float v2 = circle_img_norm[i].y;

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

        float A2 = a5 * v2 + a8;
        float A3 = a6 * v2 + a9;
        float B2 = b2 * u2 + b8;
        float B3 = b3 * u2 + b9;
        float C2 = c2 * u2 + c5 * v2;
        float C3 = c3 * u2 + c6 * v2;
        
        float D = D1 * u2 + D4 * v2 + D7;
        float R = 9.15;

        float u1 = D / (A3 * alpha + B3 * beta + C3 * gamma);
        float v1 = (A2 * alpha + B2 * beta + C2 * gamma) / (A3 * alpha + B3 * beta + C3 * gamma);
        float d = sqrt(u1 * u1 + v1 * v1) - R;

        /* float u1 = D; */
        /* float v1 = A2 * alpha + B2 * beta + C2 * gamma; */
        /* float denominator = A3 * alpha + B3 * beta + C3 * gamma; */
        /* float d = pow(u1, 2) + pow(v1, 2) - pow( (R * denominator), 2); */

        fi[i] = d;
    }
}

cv::Mat getH(float alpha, float beta, float gamma,
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

void draw_cross(cv::Point pt, cv::Mat image, int size){

        cv::Point starting1(pt.x - size, pt.y-size);
        cv::Point ending1(pt.x + size, pt.y+size);

        cv::Point starting2(pt.x + size, pt.y-size);
        cv::Point ending2(pt.x-size, pt.y+size);

        cv::Scalar line_Color(255, 0, 0);
       int thickness = 2;
       
    cv::line(image, starting1, ending1, line_Color, thickness);
    cv::line(image, starting2, ending2, line_Color, thickness);
}

void transformPoint(const cv::Point2f& input, cv::Point2f& output, const cv::Mat& H, bool isPerspective)
{
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

void mouse_callback(int  event, int  x, int  y, int  flag, void *param)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        pt.x = x;
        pt.y = y;
        new_coords = true;
    }
}

/* void getPoints(const cv::Mat& input_img, std::vector<cv::Point2f>& line_img, std::vector<cv::Point2f>& circle_img) */
/* { */
     
/* } */

void draw_line(cv::Point pt1, cv::Point pt2, cv::Mat image, cv::Scalar line_color){
    
       //Scalar line_Color(0, 0, 255);
       int thickness = 2;
       cv::line(image, pt1, pt2, line_color, thickness);
}

int main(int argc, char **argv)
{

    cv::Mat img = cv::imread("image.png", cv::IMREAD_COLOR);
    cv::namedWindow("image", cv::WINDOW_NORMAL);

    cv::setMouseCallback("image", mouse_callback);
    
    std::vector<cv::Point2f> circle_img;
    std::vector<cv::Point2f> line_img;

    int count = 0;
    char key = 0;
    while ((int)key != 27) {

        if(new_coords){
            draw_cross(pt, img, 15);
            if(count < 3){
                line_img.emplace_back(cv::Point2f(pt.x, pt.y));
                std::cout << "line_img " << pt.x << " " << pt.y << "\n";
            }
            else if (count < 10){
                circle_img.emplace_back(cv::Point2f(pt.x, pt.y));
                std::cout << "circle_img " << pt.x << " " << pt.y << "\n";
            }
            new_coords = false;

            if(count==9) break;

            count++;
        }

        imshow("image", img);
        key = cv::waitKey(1);
    }

    std::vector<cv::Point2f> line_img_norm;
    normalizeWithSyntheticCam(circle_img, circle_img_norm);
    normalizeWithSyntheticCam(line_img, line_img_norm);

    /* float circle_img_x[7] = {1239, 1180, 947, 712, 352, 763, 391}; */
    /* float circle_img_y[7] = {581, 549, 672, 681, 570, 511, 646}; */

    /* std::vector<cv::Point2f> circle_img; */
    /* for (int i = 0; i < 7; ++i) */
    /* { */
    /*     circle_img.emplace_back(cv::Point2f(circle_img_x[i], circle_img_y[i])); */
    /* } */
    
    /* normalizeWithSyntheticCam(circle_img, circle_img_norm); */

    float r = 9.15;
    float line_map_x[3]={0, 0, 0};
    float line_map_y[3]={-1*r, 0, 1*r};

    /* float line_img_x[3]={1011, 790, 510}; */
    /* float line_img_y[3]={521, 583, 668}; */

    /* std::vector<cv::Point2f> line_img; */
    std::vector<cv::Point2f> line_map;
    for(int i = 0; i < 3; ++i) 
    {
        /* line_img.emplace_back(cv::Point2f(line_img_x[i], line_img_y[i])); */
        line_map.emplace_back(cv::Point2f(line_map_x[i], line_map_y[i]));
    }

    /* std::vector<cv::Point2f> line_img_norm; */
    /* normalizeWithSyntheticCam(line_img, line_img_norm); */

    cv::Mat H = calcHomography(line_map, line_img_norm);

    h2=H.at<float>(0, 1);
    h3=H.at<float>(0, 2);
    h5=H.at<float>(1, 1);
    h6=H.at<float>(1, 2);
    h8=H.at<float>(2, 1);

    real_1d_array x = "[0.0001,0.0001,0.0001]";
    /* real_1d_array x = "[0.0,0.0,0.0]"; */
    real_1d_array s = "[1,1,1]";

    double epsx = 0;
    ae_int_t maxits = 0;
    minlmstate state;
    minlmreport rep;

    minlmcreatev(3, 7, x, 0.0001, state);
    minlmsetcond(state, epsx, maxits);
    minlmsetscale(state, s);

    alglib::minlmoptimize(state, function1_fvec);

    minlmresults(state, x, rep);
    printf("%s\n", x.tostring(8).c_str()); 

    /* float H_data[9] = {500, 0, 600, 0, 500, 400, 0, 0, 1}; */
    /* cv::Mat H = cv::Mat(3, 3, CV_32F, H_data); */
    /* H.at<float>(0, 0) = x[0]; */
    /* H.at<float>(1, 0) = x[1]; */
    /* H.at<float>(2, 0) = x[2]; */

    float alpha = x[0];
    float beta = x[1];
    float gamma = x[2];

    float K_data[9] = { 500, 0, 600, 0, 500, 400, 0, 0, 1 };
    cv::Mat K = cv::Mat(3, 3, CV_32F, K_data);

    /* cv::Mat_<float> K(3,3); // rows, cols */
    /* K << 500, 0, 600, 0, 500, 400, 0, 0, 1; */

    cv::Mat final_H = getH(alpha, beta, gamma, h2, h3, h5, h6, h8);

    cv::Mat ans = final_H * K.inv();
    cv::Mat inv_final_H = ans.inv();

    /* inv_final_H = inv_final_H * (1.0 / inv_final_H.at<float>(2, 2)); */

    std::cout << inv_final_H << "\n";

    /* generate points */
    std::vector<cv::Point2f> points_map;
    points_map.emplace_back(cv::Point2f(0, -1*r));
    points_map.emplace_back(cv::Point2f(0, 0));
    points_map.emplace_back(cv::Point2f(0, 1*r));
    
    /* generating unit cirlce for the map */
    for(int i = 1; i < 360; i++){
        float angle = static_cast<float>(i) * 3.14159 / 180.0;
        float x = r * cos(angle);
        float y = r * sin(angle);
        points_map.emplace_back(cv::Point2f(x, y));
    }


    for(auto & e: points_map){
        cv::Point2f output;
        transformPoint(e, output, inv_final_H, true);
        draw_cross(output, img, 3);
    }

    while ((int)key != 27) {
        if(new_coords){
            cv::Point2f line_a;
            cv::Point2f line_b;
            cv::Point2f transformed_pt;

            transformPoint(pt, transformed_pt, ans, true);
            transformPoint(cv::Point2f(transformed_pt.x, 40), line_a, inv_final_H, true);
            transformPoint(cv::Point2f(transformed_pt.x, -40), line_b, inv_final_H, true);

            draw_line(line_a, line_b, img, cv::Scalar(0, 0, 255));
            new_coords = false;
        }
        imshow("image", img);
        key = cv::waitKey(1);
    }
    return 0;
}
