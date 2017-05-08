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

    void normalizeFeatures();

    Mat calculateCovarianceMatrix();

    Mat transformInputValues(Mat eigenVectors);

public:
    PrincipalComponentAnalysis();

    Mat performPCA(int count);

    void addFruitData(vector<double> fruitData);

};

#endif //FRUITRECOGNITION_PRINCIPALCOMPONENTANALYSIS_H
