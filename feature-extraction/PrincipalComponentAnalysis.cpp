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
    for (int row = 0; row < fruitGroup.rows; row++) {
        Scalar mean, standardDeviation;
        meanStdDev(fruitGroup.row(row), mean, standardDeviation);
        for (int col = 0; col < fruitGroup.cols; col++) {
            fruitGroup.at<double>(row, col) -= mean[0];
//            fruitGroup.at<double>(row, col) /= standardDeviation[0];
        }
    }
}

void PrincipalComponentAnalysis::addFruitData(vector<double> fruitData) {
    if (fruitGroup.cols == 0) {
        fruitGroup = Mat(fruitData.size(), 0, CV_64F);
        hconcat(fruitGroup, Mat(fruitData), fruitGroup);
    } else {
        hconcat(fruitGroup, Mat(fruitData), fruitGroup);
    }
}

Mat PrincipalComponentAnalysis::calculateCovarianceMatrix() {
    cv::Mat mean, covs;
    cv::calcCovarMatrix(fruitGroup, covs, mean, CV_COVAR_NORMAL | CV_COVAR_COLS);
    covs = covs / (fruitGroup.cols - 1);
    return covs;
}

Mat PrincipalComponentAnalysis::transformInputValues(Mat eigenVectors) {
    Mat result = eigenVectors.t() * fruitGroup;
    return result;
}

