//
// Created by Long Bui on 04.05.17.
//
#ifndef FRUITRECOGNITION_PRINCIPALCOMPONENTANALYSIS_H
#define FRUITRECOGNITION_PRINCIPALCOMPONENTANALYSIS_H

#include <iostream>
#include <opencv2/core.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigenvalues>

using namespace std;
using namespace cv;

class PrincipalComponentAnalysis {
private:
    Mat fruitGroup;

    void normalizeValues(vector<double> &values);

    Mat calculateCovarianceMatrix();

public:
    PrincipalComponentAnalysis();

    Mat performPCA(int count);

    void addFruitFeatures(vector<double> fruitFeatures);
};

#endif //FRUITRECOGNITION_PRINCIPALCOMPONENTANALYSIS_H
