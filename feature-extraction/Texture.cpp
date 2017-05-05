//
// Created by Long Bui on 25.04.17.
//

#include "Texture.h"

/*
 * fill the 256-sized histogram array:
 * 1. For every cell go the whole image through (polnish method)
 * 2. or go the image only 1 time and fill the h-array by increments.
 * 3. compare
 */

//optimal calculation of histogram with O(n^2), not O(n^3)
vector<int> histogramOpt(const Mat &grayImage) {
    vector<int> hist(256, 0);
    for (int row = 0; row < grayImage.rows; row++)
        for (int col = 0; col < grayImage.cols; col++)
            hist[(int)grayImage.at<uchar>(row, col)]++;
    return hist;
}

//O(n^3)
vector<int> histogramLong(const Mat &grayImage) {
    vector<int> hist(256, 0);
    for (int i = 0; i < hist.size(); ++i)
        for (int row = 0; row < grayImage.rows; row++)
            for (int col = 0; col < grayImage.cols; col++)
                if(i == (int)grayImage.at<uchar>(row, col))
                    hist[i]++;
    return hist;
}

void histogramCompare(const Mat &grayImage)
{
    vector<int> h1 = histogramOpt(grayImage);
    vector<int> h2 = histogramLong(grayImage);
    if(h1.size() != h2.size())
        cout << "FALSE: different sizes" << endl;
    else
    {
        bool right = 1;
        for (int i = 0; i < h1.size(); ++i)
            if(h1[i] != h2[i]) {
                cout << "FALSE: wrong element nr. " << i << endl;
                right = 0;
                break;
            }
        if(right)
            cout << "Okaaay, true.." << endl;
    }
}

//"approximate probability density of occurrence of the intensity levels"
//Zitat aus dem polnischen Paper
vector<double> probability(const vector<int> hist, int size) {
    vector<double> p(hist.size());
    for (int i = 0; i < hist.size(); i++)
        p[i] = ((double) hist[i]) / ((double) size);
    return p;
}

// = = = = BUG IS POSSIBLE = = = =
//in Paper wird mean durch h_sum berechnet
double mean(const vector<double> &p) {
    double sum = 0;
    for (int i = 0; i < p.size(); i++)
        sum += i*p[i];
    return sum;
}

// = = = = BUG IS POSSIBLE = = = =
//in Paper wird contrast durch h_diff berechnet
double contrast(const vector<double> &p) {
    double sum = 0;
    for (int i = 0; i < p.size(); i++)
        sum += i*i*p[i];
    return sum;
}

//ebenso durch h_diff
double homogenity(const vector<double> &p) {
    double sum = 0;
    for (int i = 0; i < p.size(); i++)
        sum += p[i]/(1+i*i);
    return sum;
}

double variance(const vector<double> &p, double mean) {
    double sum = 0;
    for (int i = 0; i < p.size(); i++)
        sum += pow(i - mean, 2) * p[i];
    return pow(sum, 0.5);
}

double skewness(const vector<double> &p, double mean, double variance) {
    double sum = 0;
    for (int i = 0; i < p.size(); ++i)
        sum += pow(i - mean, 3) * p[i];
    return sum * pow(variance, -3);
}

//picture flatness
double kurtosis(const vector<double> &p, double mean, double variance) {
    double sum = 0;
    for (int i = 0; i < p.size(); ++i)
        sum += pow(i - mean, 4) * p[i];
    return sum * pow(variance, -4) - 3;
}

double energy(const vector<double> &p) {
    double sum = 0;
    for (int i = 0; i < p.size(); ++i)
        sum += pow(p[i], 2);
    return sum;
}

double entropy(const vector<double> &p) {
    double sum = 0;
    for (int i = 0; i < p.size(); ++i)
        if(p[i] > 0)
            sum += p[i] * log2(p[i]);
    return -sum;
}

vector<double> unser(const Mat &grayImage)
{
    vector<double> probabilities = probability(histogramOpt(grayImage), grayImage.cols*grayImage.rows);
    double meanValue = mean(probabilities);
    double varianceValue = variance(probabilities, meanValue);
    vector<double> results;
    results.push_back(meanValue);
    results.push_back(varianceValue);
    results.push_back(contrast(probabilities)); //Paper
    results.push_back(homogenity(probabilities)); //Paper
    results.push_back(skewness(probabilities, meanValue, varianceValue)); //PL
    results.push_back(kurtosis(probabilities, meanValue, varianceValue)); //PL
    results.push_back(energy(probabilities));
    results.push_back(entropy(probabilities));
    return results;
}

vector<double> unserHist(const Mat &grayImage)
{
    vector<int> histInt = histogramOpt(grayImage);
    vector<double> histDouble (histInt.size());
    for (int i = 0; i < histDouble.size(); i++)
        histDouble[i] = (double)histInt[i];
    //vector<double> probabilities = probability(histogramOpt(grayImage), grayImage.cols*grayImage.rows);
    double meanValue = mean(histDouble);
    double varianceValue = variance(histDouble, meanValue);
    vector<double> results;
    results.push_back(meanValue);
    results.push_back(varianceValue);
    results.push_back(contrast(histDouble)); //Paper
    results.push_back(homogenity(histDouble)); //Paper
    results.push_back(skewness(histDouble, meanValue, varianceValue)); //PL
    results.push_back(kurtosis(histDouble, meanValue, varianceValue)); //PL
    results.push_back(energy(histDouble));
    results.push_back(entropy(histDouble));
    return results;
}

void unserTest(const Mat &grayImage){
    //berechnet alle Werte anhand Wahrcsheinlichkeiten
    vector<double> textures = unser(grayImage);
    //berechnet alle Werte nur durch Histogram. Ergebnis sieht komisch aus
    vector<double> texturesHist = unserHist(grayImage);
    cout << "\t= = = Data block = = =" << endl;
    for (int i = 0; i < textures.size(); ++i)
        cout << textures[i] << "\t\t" << texturesHist[i] << endl;
}