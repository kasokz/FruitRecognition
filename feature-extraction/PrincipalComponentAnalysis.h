//
// Created by Long Bui on 04.05.17.
//
#ifndef FRUITRECOGNITION_PRINCIPALCOMPONENTANALYSIS_H
#define FRUITRECOGNITION_PRINCIPALCOMPONENTANALYSIS_H

#include <iostream>
#include <opencv2/core.hpp>
#include <eigen3/Eigen/dense>

using namespace std;
using namespace cv;
using namespace Eigen;

class PrincipalComponentAnalysis {
private:
    Mat eigenvalues, eigenvectors, principalComponents;

    Scalar mean, standardDeviation;

    void fitNormalization(Mat fruitFeatures);

    Mat calculateCovarianceMatrix(Mat &dataset);

public:
    PrincipalComponentAnalysis();

    void fit(const Mat &fruitFeatures, int count);

    void normalize(Mat &data);

    Mat project(const Mat &data);

    Mat backProject(const Mat &data);

    Mat getEigenvalues();

    Mat getEigenvectors();
};

#endif //FRUITRECOGNITION_PRINCIPALCOMPONENTANALYSIS_H
