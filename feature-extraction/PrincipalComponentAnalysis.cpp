//
// Created by Long Bui on 04.05.17.
//

#include "PrincipalComponentAnalysis.h"

PrincipalComponentAnalysis::PrincipalComponentAnalysis() {
}

Mat PrincipalComponentAnalysis::performPCA(int count) {
    normalizeFeatures();
    Mat covarianceMatrix = calculateCovarianceMatrix();
    Mat eigenValues, eigenVectors;
    eigen(covarianceMatrix, eigenValues, eigenVectors);
    Mat principalComponents = Mat(eigenVectors, Range(0, eigenVectors.rows), Range(0, count));
    return transformInputValues(principalComponents);
}

void PrincipalComponentAnalysis::normalizeFeatures() {
    Scalar mean, standardDeviation;
    meanStdDev(fruitFeatures, mean, standardDeviation);
    for (int row = 0; row < fruitFeatures.rows; row++) {
        for (int col = 0; col < fruitFeatures.cols; col++) {
            fruitFeatures.at<double>(row, col) -= mean[0];
            fruitFeatures.at<double>(row, col) /= standardDeviation[0];
        }
    }
}

void PrincipalComponentAnalysis::addFruitData(vector<double> fruitData) {
    if (fruitFeatures.cols == 0) {
        fruitFeatures = Mat(fruitData.size(), 1, CV_64F, fruitData.data());
    } else {
        hconcat(fruitFeatures, Mat(fruitData), fruitFeatures);
    }
}

Mat PrincipalComponentAnalysis::calculateCovarianceMatrix() {
    cv::Mat mean, covs;
    cv::calcCovarMatrix(fruitFeatures, covs, mean, CV_COVAR_NORMAL | CV_COVAR_COLS);
    covs = covs / (fruitFeatures.cols - 1);
    return covs;
}

Mat PrincipalComponentAnalysis::transformInputValues(Mat eigenVectors) {
    Mat result = eigenVectors.t() * fruitFeatures;
    return result;
}

