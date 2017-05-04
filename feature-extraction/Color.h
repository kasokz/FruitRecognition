//
// Created by Long Bui on 25.04.17.
//
#ifndef FRUITRECOGNITION_COLOR_H_H
#define FRUITRECOGNITION_COLOR_H_H


#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

vector<unsigned int> extractColorHistogram(Mat &image);

#endif //FRUITRECOGNITION_COLOR_H_H
