//
// Created by Long Bui on 04.05.17.
//

#include "PrincipalComponentAnalysis.h"

PrincipalComponentAnalysis::PrincipalComponentAnalysis() {
}

Mat PrincipalComponentAnalysis::calculateCovarianceMatrix(Mat &dataset) {
    cv::Mat mean, covs;
    cv::calcCovarMatrix(dataset, covs, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);
    covs = covs / (dataset.cols - 1);
    return covs;
}

void PrincipalComponentAnalysis::fitNormalization(Mat fruitFeatures) {
    meanStdDev(fruitFeatures, mean, standardDeviation);
}

void PrincipalComponentAnalysis::normalize(Mat &data) {
    for (int row = 0; row < data.rows; row++) {
        for (int col = 0; col < data.cols; col++) {
            data.at<double>(row, col) -= mean[0];
            data.at<double>(row, col) /= standardDeviation[0];
        }
    }
}

void PrincipalComponentAnalysis::fit(const Mat &fruitFeatures, int count) {
    Mat copy;
    fruitFeatures.copyTo(copy);
    fitNormalization(fruitFeatures);
    normalize(copy);
    Mat covarianceMatrix = calculateCovarianceMatrix(copy);
    eigen(covarianceMatrix, this->eigenvalues, this->eigenvectors);
    this->principalComponents = Mat(this->eigenvectors, Range(0, count), Range(0, this->eigenvectors.rows));
}

Mat PrincipalComponentAnalysis::project(const Mat &data) {
    Mat normalizedData;
    data.copyTo(normalizedData);
    normalize(normalizedData);
    return this->principalComponents * normalizedData.t();
}

Mat PrincipalComponentAnalysis::backProject(const Mat &data) {
    Mat inverted;
    invert(this->principalComponents, inverted, DECOMP_SVD);
    Mat original = inverted * data;
    original *= standardDeviation[0];
    original += mean[0];
    return original;
}

Mat PrincipalComponentAnalysis::getEigenvalues() {
    return eigenvalues;
}

Mat PrincipalComponentAnalysis::getEigenvectors() {
    return eigenvectors;
}
