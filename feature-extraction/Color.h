//
// Created by Long Bui on 25.04.17.
//
#ifndef FRUITRECOGNITION_COLOR_H
#define FRUITRECOGNITION_COLOR_H

#include <opencv2/opencv.hpp>
using namespace cv;

std::vector<double> extractColorHistogram(cv::Mat &image);

#endif //FRUITRECOGNITION_COLOR_H
