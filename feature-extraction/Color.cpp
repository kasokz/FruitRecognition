//
// Created by Long Bui on 25.04.17.
//
#include "Color.h"

std::vector<double> extractColorHistogram(cv::Mat &image) {
    std::vector<double> histogram(64, 0);
    for (int row = 0; row < image.rows; row++) {
        for (int col = 0; col < image.cols; col++) {
            cv::Vec3b &pixel = image.at<cv::Vec3b>(row, col);
            double blue = pixel[0] / 64;
            double green = pixel[1] / 64;
            double red = pixel[2] / 64;
            histogram[blue * 16 + green * 4 + red]++;
        }
    }
    histogram[63] = 0;
    return histogram;
}