//
// Created by Long Bui on 04.05.17.
//
#ifndef FRUITRECOGNITION_PRINCIPALCOMPONENTANALYSIS_H
#define FRUITRECOGNITION_PRINCIPALCOMPONENTANALYSIS_H

#include <iostream>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

class PrincipalComponentAnalysis {
private:
    Mat fruitGroup;

public:
    PrincipalComponentAnalysis();

    vector<double> performPCA(vector<double> input, int count);

    void addFruitFeatures(vector<double> fruitFeatures);

    void normalizeValues(vector<double, allocator<double>> *values);
};

#endif //FRUITRECOGNITION_PRINCIPALCOMPONENTANALYSIS_H
