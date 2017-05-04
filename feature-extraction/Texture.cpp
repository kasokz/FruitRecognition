//
// Created by Long Bui on 25.04.17.
//

#include "Texture.h"

/*
 * fill the 256-sized histogram array:
 * 1. For every cell go the whole image through (polnish method)
 * 2. or go the image only 1 time and fill the h-array by increments.
 * 3. compare
 */

//optimal calculation of histogram with O(2), not O(3)
vector<int> histogramOpt(const Mat &grayImage) {
    vector<int> hist(256, 0);
    for (int row = 0; row < grayImage.rows; ++row)
        for (int col = 0; col < grayImage.cols; ++col)
            hist[(int)grayImage.at<uchar>(row, col)]++;
    return hist;
}

//approximate probability density of occurrence of the intensity levels
vector<double> probability(const vector<int> hist, int size) {
    vector<double> p(hist.size());
    for (int i = 0; i < hist.size(); i++)
        p[i] = ((double)hist[i])/((double)size);
    return p;
}

double mean(const vector<double> p) {
    double sum = 0;
    for (int i = 0; i < p.size(); i++)
        sum += i*p[i];
    return sum;
}

double variance(const vector<double> p, double mean) {
    double sum = 0;
    for (int i = 0; i < p.size(); i++)
        sum += pow(i - mean, 2) * p[i];
    return pow(sum, 0.5);
}

double skewness(const vector<int> p, double mean, double variance) {
    double sum = 0;
    for (int i = 0; i < p.size(); ++i)
        sum += pow(i - mean, 3) * p[i];
    return sum * pow(variance, -3);
}

//eventuell Kurtosis

double energy(const vector<int> p) {
    double sum = 0;
    for (int i = 0; i < p.size(); ++i)
        sum += pow(p[i], 2);
}

double entropy(const vector<int> p) {
    double sum = 0;
    for (int i = 0; i < p.size(); ++i)
        sum += p[i] * log2(p[i]);
    return -sum;
}