//
// Created by Long Bui on 04.05.17.
//

#include "PrincipalComponentAnalysis.h"

vector<double> PrincipalComponentAnalysis::performPCA(vector<double> input, int count) {
    vector<double> result((unsigned long) count, 0);
    normalizeValues(&input);

    return result;
}

void PrincipalComponentAnalysis::normalizeValues(vector<double, allocator<double>> *values) {
    Scalar mean, standardDeviation;
    meanStdDev(*values, mean, standardDeviation);
    for (int i = 0; i < values->size(); i++) {
        (*values)[i] -= mean[0];
        (*values)[i] /= standardDeviation[0];
    }
}

PrincipalComponentAnalysis::PrincipalComponentAnalysis() {

}

void PrincipalComponentAnalysis::addFruitFeatures(vector<double> fruitFeatures) {
    hconcat(this->fruitGroup, fruitFeatures);
}
