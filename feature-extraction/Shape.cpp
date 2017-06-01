//
// Created by Long Bui on 25.04.17.
//

#include "Shape.h"

int area(const Mat &image)
{
    int sum = 0;
    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
            if(image.at<uchar>(i,j) != 255)
                sum++;
    return sum;
}

//int

vector<double> shape(const Mat &image) {
    vector<double> results;

    return results;
}