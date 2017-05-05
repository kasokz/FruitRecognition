//
// Created by Long Bui on 04.05.17.
//

#include "PrincipalComponentAnalysis.h"

PrincipalComponentAnalysis::PrincipalComponentAnalysis() {

}

Mat PrincipalComponentAnalysis::performPCA(int count) {
    Mat result;
    Mat covarianceMatrix = calculateCovarianceMatrix();
    Mat eigenValues, eigenVectors;
    eigen(covarianceMatrix, eigenValues, eigenVectors);
    cout << eigenVectors << endl;
    return result;
}

void PrincipalComponentAnalysis::normalizeValues(vector<double> &values) {
    Scalar mean, standardDeviation;
    meanStdDev(values, mean, standardDeviation);
    for (int i = 0; i < values.size(); i++) {
        values[i] -= mean[0];
        values[i] /= standardDeviation[0];
    }
}

void PrincipalComponentAnalysis::addFruitFeatures(vector<double> fruitFeatures) {
    normalizeValues(fruitFeatures);
    if (this->fruitGroup.cols == 0) {
        this->fruitGroup = Mat(fruitFeatures);
    } else {
        hconcat(this->fruitGroup, Mat(fruitFeatures), this->fruitGroup);
    }
}

Mat PrincipalComponentAnalysis::calculateCovarianceMatrix() {
    cv::Mat mean, covs;
    cv::calcCovarMatrix(fruitGroup, covs, mean, CV_COVAR_NORMAL | CV_COVAR_COLS);
    covs = covs / (fruitGroup.rows - 1);
    return covs;
}
