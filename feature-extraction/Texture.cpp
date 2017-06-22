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

//O(n^3) ... NEVER USER, just for test
vector<int> histogramLong(const Mat &grayImage) {
    vector<int> hist(256, 0);
    for (int i = 0; i < hist.size(); ++i)
        for (int row = 0; row < grayImage.rows; row++)
            for (int col = 0; col < grayImage.cols; col++)
                if(i == (int)grayImage.at<uchar>(row, col))
                    hist[i]++;
    return hist;
}
//to compare 2 ways of histogram calculating O(n^2) and O(n^3) ... NEVER USED
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

// = = = = MATH BUG IS POSSIBLE = = = =
//in Paper wird mean durch h_sum berechnet
double mean(const vector<double> &p) {
    double sum = 0;
    for (int i = 0; i < p.size(); i++)
        sum += i*p[i];
    return sum;
}

// = = = = MATH BUG IS POSSIBLE = = = =
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

// = = = = = Advanced features = = = = =

Mat co_occurrence(const Mat &grayImage, int angle)
{
    if(angle != 0 && angle != 90)
        angle = 0;
    Mat output = Mat::zeros(256, 256, CV_32S);
    int horizontal = angle == 0 ? 1 : 0;
    int vertical = angle == 90 ? 1 : 0;
    for (int i = 0; i < grayImage.rows - vertical; ++i) {
        for (int j = 0; j < grayImage.cols - horizontal; ++j) {
            output.at<int>(grayImage.at<uchar>(i, j), grayImage.at<uchar>(i+vertical, j+horizontal))++;
            output.at<int>(grayImage.at<uchar>(i+vertical, j+horizontal), grayImage.at<uchar>(i, j))++;
        }
    }
    return output;
}

//test method NEVER USED
void co_occurrenceTest()
{
    vector<int> arr = {0,0,1,1,0,0,1,1,0,2,2,2,2,2,3,3};
    Mat in = Mat(4, 4, CV_32S, arr.data());
    Mat out = co_occurrence(in);
    cout << in << endl << endl << out << endl << endl;
}

double energy2(const Mat &mat)
{
    double sum = 0;
    for (int i = 0; i < mat.rows; ++i)
        for (int j = 0; j < mat.cols; ++j)
            sum += pow(mat.at<int>(i, j), 2);
    return sum;
}

double correlation(const Mat &mat)
{
    vector<double> meanX, meanY, staDevX, staDevY;
    double sum;
    for (int i = 0; i < mat.rows; ++i) {
        sum = 0;
        for (int j = 0; j < mat.cols; ++j)
            sum += mat.at<int>(i, j);
        meanX.push_back(sum/mat.cols);
    }
    for (int j = 0; j < mat.cols; ++j) {
        sum = 0;
        for (int i = 0; i < mat.rows; ++i)
            sum += mat.at<int>(i, j);
        meanY.push_back(sum/mat.rows);
    }
    for (int i = 0; i < mat.rows; ++i) {
        sum = 0;
        for (int j = 0; j < mat.cols; ++j)
            sum += pow(mat.at<int>(i, j)-meanX[i], 2);
        staDevX.push_back(sqrt(sum/mat.cols));
    }
    for (int j = 0; j < mat.cols; ++j) {
        sum = 0;
        for (int i = 0; i < mat.rows; ++i)
            sum += pow(mat.at<int>(i, j) - meanY[j], 2);
        staDevY.push_back(sqrt(sum/mat.rows));
    }
    sum = 0;
    for (int i = 0; i < mat.rows; ++i)
        for (int j = 0; j < mat.cols; ++j)
            if(staDevX[j] != 0 && staDevY[i] != 0)
                sum += (i*j*mat.at<int>(i, j) - meanX[j]*meanY[i]) / (staDevX[j]*staDevY[i]);
    return sum;
}

double interia(const Mat &mat)
{
    double sum = 0;
    for (int i = 0; i < mat.rows; ++i)
        for (int j = 0; j < mat.cols; ++j)
            sum += pow(i-j,2)*mat.at<int>(i,j);
    return sum;
}

double absoluteValue(const Mat &mat)
{
    double sum = 0;
    for (int i = 0; i < mat.rows; ++i)
        for (int j = 0; j < mat.cols; ++j)
            sum += abs(i-j)*mat.at<int>(i,j);
    return sum;
}

double inverseDifference(const Mat &mat)
{
    double sum = 0;
    for (int i = 0; i < mat.rows; ++i)
        for (int j = 0; j < mat.cols; ++j)
            sum += mat.at<int>(i,j)/(1+pow(i-j,2));
    return sum;
}

double entropy2(const Mat &mat)
{
    double sum = 0;
    for (int i = 0; i < mat.rows; ++i)
        for (int j = 0; j < mat.cols; ++j)
            if(mat.at<int>(i,j) > 0)
                sum += mat.at<int>(i,j) * log2(mat.at<int>(i,j));
    return -sum;
}

int maxP(const Mat &mat)
{
    int max = -1;
    for (int i = 0; i < mat.rows; ++i)
        for (int j = 0; j < mat.cols; ++j)
            if(mat.at<int>(i,j) > max)
                max = mat.at<int>(i,j);
    return max;
}

vector<double> unser(const Mat &grayImage)
{
    vector<double> probabilities = probability(histogramOpt(grayImage), grayImage.cols*grayImage.rows);
    double meanValue = mean(probabilities);
    double varianceValue = variance(probabilities, meanValue);
    vector<double> results;
    results.push_back(meanValue);
    results.push_back(varianceValue);
    results.push_back(contrast(probabilities)); //Orig Paper
    results.push_back(homogenity(probabilities)); //Orig Paper
    results.push_back(skewness(probabilities, meanValue, varianceValue)); //PL
    results.push_back(kurtosis(probabilities, meanValue, varianceValue)); //PL
    results.push_back(energy(probabilities));
    results.push_back(entropy(probabilities));
    Mat co_occurrenceMatrix = co_occurrence(grayImage);
//    results.push_back(energy2(co_occurrenceMatrix));
//    results.push_back(correlation(co_occurrenceMatrix));
//    results.push_back(interia(co_occurrenceMatrix));
    results.push_back(absoluteValue(co_occurrenceMatrix));
    results.push_back(inverseDifference(co_occurrenceMatrix));
//    results.push_back(entropy2(co_occurrenceMatrix));
//    results.push_back(maxP(co_occurrenceMatrix));
    return results;
}
//another test NEVER USED
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