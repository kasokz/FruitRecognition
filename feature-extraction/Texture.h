//
// Created by Long Bui on 25.04.17.
//

#ifndef FRUITRECOGNITION_TEXTURE_H
#define FRUITRECOGNITION_TEXTURE_H

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

vector<int> histogramOpt(const Mat &grayImage);
vector<int> histogramLong(const Mat &grayImage);
void histogramCompare(const Mat &grayImage);
vector<double> probability(const vector<int> hist, int size);
double mean(const vector<double> &p);
double contrast(const vector<double> &p);
double homogenity(const vector<double> &p);
double variance(const vector<double> &p, double mean);
double skewness(const vector<double> &p, double mean, double variance);
double kurtosis(const vector<double> &p, double mean, double variance);
double energy(const vector<double> &p);
double entropy(const vector<double> &p);
Mat co_occurrence(const Mat &grayImage, int angle = 0);
double correlation(const Mat &mat);
void co_occurrenceTest();
vector<double> unser(const Mat &grayImage);
vector<double> unserHist(const Mat &grayImage);
void unserTest(const Mat &grayImage);

#endif //FRUITRECOGNITION_TEXTURE_H