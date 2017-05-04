//
// Created by Long Bui on 25.04.17.
//
#include "Color.h"

std::vector<double> extractColorHistogram(cv::Mat &image) {
    std::vector<double> histogram(64, 0);
    for (int row = 0; row < image.rows; row++) {
        for (int col = 0; col < image.cols; col++) {
            cv::Vec3b &pixel = image.at<cv::Vec3b>(row, col);
            pixel[0] /= 64;
            pixel[1] /= 64;
            pixel[2] /= 64;
            histogram[pixel[0] * 16 + pixel[1] * 4 + pixel[2]]++;
        }
    }
    histogram[63] = 0;
    return histogram;
}