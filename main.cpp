#include <iostream>
#include <opencv2/opencv.hpp>
#include "segmentation/Quadtree.h"
#include "feature-extraction/Color.h"
#include "feature-extraction/Texture.h"
#include "feature-extraction/PrincipalComponentAnalysis.h"

using namespace cv;
using namespace std;

map<String, vector<String>> getAllFileNames();

void readCsvDataset(vector<vector<double, allocator<double>>, allocator<vector<double, allocator<double>>>> &all_data,
                    vector<string, allocator<string>> &responses);

Mat convertToMat(vector<vector<double>> data);

void predictTrainingData(const Mat &reducedFeatures, Mat &responseIndices, const Ptr<ml::SVM> &svm);

vector<double> extractFeatures(Mat &rgbImage);

void segmentImage(Mat &rgbImage, const Mat &thresholdImage);

Ptr<ml::SVM> createAndTrainSvm(Mat features, Mat responses);

String fruits[] = {
        "apples", "apricots", "avocados", "bananas", "blackberries", "blueberries", "cantaloupes", "cherries",
        "coconuts", "figs", "grapefruits", "grapes", "guava", "kiwifruit", "lemons", "limes", "mangos", "olives",
        "oranges", "passionfruit", "peaches", "pears", "pineapples", "plums", "pomegranates", "raspberries",
        "strawberries", "tomatoes", "watermelons"};

int getIndexOfFruit(string fruitname) {
    int i = 0;
    for (String s: fruits) {
        if (s == fruitname) {
            break;
        } else {
            i++;
        }
    }
    return i;
}

void segmentImage(Mat &rgbImage, const Mat &thresholdImage) {
    for (int row = 0; row < rgbImage.rows; ++row) {
        for (int col = 0; col < rgbImage.cols; ++col) {
            if (!thresholdImage.at<uchar>(row, col)) {
                Vec3b *pixel = rgbImage.ptr<Vec3b>(row, col);
                (*pixel)[0] = 255;
                (*pixel)[1] = 255;
                (*pixel)[2] = 255;
            }
        }
    }
}

map<String, vector<String>> getAllFileNames() {
    map<String, vector<String>> filenames;
    String folder = "FIDS30/";

    for (String fruit: fruits) {
        vector<String> filesForFruit;
        glob(folder + fruit, filesForFruit);
        filenames[fruit] = filesForFruit;
    }
    return filenames;
}

void fillHolesInThreshold(const Mat &grayImage, Mat &thresholdImage) {
    threshold(grayImage, thresholdImage, 0, 255,
              CV_THRESH_BINARY_INV + CV_THRESH_OTSU);

    // Floodfill from point (0, 0)
    Mat im_floodfill = thresholdImage.clone();
    floodFill(im_floodfill, Point(0, 0), Scalar(255));

    // Invert floodfilled image
    bitwise_not(im_floodfill, im_floodfill);

    // Combine the two images to get the foreground.
    thresholdImage = (thresholdImage | im_floodfill);
}

void createDatasetAsCsv() {
    ofstream csvFile;
    csvFile.open("fruit_features.csv");
    for (int i = 0; i < 64; i++) {
        csvFile << "color" << i << ",";
    }
    for (int i = 0; i < 8; i++) {
        csvFile << "unser" << i << ",";
    }
//    for (int i = 0; i < 8; i++) {
//        csvFile << "shape" << i << ",";
//    }
    csvFile << "class" << endl;

    map<String, vector<String>> filenames = getAllFileNames();
    for (String fruit: fruits) {
        for (String fruitFile: filenames[fruit]) {
            Mat rgbImage;
            rgbImage = imread(fruitFile);
            if (!rgbImage.data) {
                cout << fruitFile << endl;
                printf("No image data \n");
            } else {
                vector<double> extractedFeatures = extractFeatures(rgbImage);
                Mat features = Mat(1, (int) extractedFeatures.size(), CV_64F, extractedFeatures.data());
                csvFile << cv::format(features, cv::Formatter::FMT_CSV);
                csvFile << "," << fruit << endl;
//                unserTest(grayImage);
            }
        }
    }
    csvFile.close();
}

vector<double> extractFeatures(Mat &rgbImage) {
    Mat grayImage;
    cvtColor(rgbImage, grayImage, COLOR_RGB2GRAY);

    shared_ptr<Quadtree> root(new Quadtree(grayImage));
    root->splitAndMerge();

    Mat thresholdImage;
    fillHolesInThreshold(grayImage, thresholdImage);

    segmentImage(rgbImage, thresholdImage);
    resize(rgbImage, rgbImage, Size(256, 256));
    resize(grayImage, grayImage, Size(256, 256));

    vector<double> extractedFeatures = Mat(extractColorHistogram(rgbImage));
    vector<double> textures = Mat(unser(grayImage));
    extractedFeatures.insert(extractedFeatures.end(), textures.begin(), textures.end());
    return extractedFeatures;
}

vector<vector<double>> readCsvDataset(vector<string> &responses) {
    vector<vector<double>> all_data;
    ifstream inputfile("fruit_features.csv");
    string current_line;
    while (getline(inputfile, current_line)) {
        vector<double> values;
        stringstream temp(current_line);
        string single_value;
        int count = 0;
        while (getline(temp, single_value, ',')) {
            if (count == 72) {
                responses.push_back(single_value.c_str());
                count++;
            } else {
                values.push_back(atof(single_value.c_str()));
                count++;
            }
        }
        all_data.push_back(values);
    }
    all_data.erase(all_data.begin());
    responses.erase(responses.begin());
    return all_data;
}

void writeFeaturesToCsv(const Mat &reducedFeatures, const Mat &responseIndices) {
    ofstream csvFile;
    csvFile.open("pca_features.csv");
    csvFile << "c1,c2,c3,class" << endl;
    for (int i = 0; i < reducedFeatures.rows; i++) {
        csvFile << cv::format(reducedFeatures.row(i), cv::Formatter::FMT_CSV);
        csvFile << "," << responseIndices.at<int>(i, 0) << endl;
    }
}

void predictTrainingData(const Mat &reducedFeatures, Mat &responseIndices, const Ptr<ml::SVM> &svm) {
    Mat result;
    svm->predict(reducedFeatures.t(), result);

    int correctPredictions = 0;
    int sumPredictions = 0;
    for (int i = 0; i < responseIndices.rows; i++) {
        cout << "Predicted: " << result.at<float>(i) << ", Actual: " << responseIndices.at<int>(i) << endl;
        sumPredictions++;
        if (result.at<float>(i) == responseIndices.at<int>(i)) {
            correctPredictions++;
        }
    }
    cout << "Correct Predictions: " << correctPredictions << "(" << setprecision(4)
         << ((double) correctPredictions / sumPredictions) * 100 << "%)" << endl;
}

Mat convertToMat(vector<vector<double>> data) {
    Mat result((int) data.size(), (int) data[0].size(), CV_64F);
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[i].size(); j++) {
            result.at<double>(i, j) = data[i][j];
        }
    }
    return result;
}

Ptr<ml::SVM> createAndTrainSvm(Mat features, Mat responses) {
    Ptr<ml::TrainData> trainData = ml::TrainData::create(features, ml::SampleTypes::COL_SAMPLE, responses);
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::RBF);
    svm->trainAuto(trainData, 5);
    return svm;
}

int main(int argc, char **argv) {
//    createDatasetAsCsv();
    vector<string, allocator<string>> responses;
    vector<vector<double>> all_data = readCsvDataset(responses);

    shared_ptr<PrincipalComponentAnalysis> pca(new PrincipalComponentAnalysis());
    for (vector<double> a: all_data) {
        pca->addFruitData(a);
    }
    pca->fit(14);
    Mat dataAsMat = convertToMat(all_data);
    Mat reducedFeatures = pca->project(dataAsMat);

    reducedFeatures.convertTo(reducedFeatures, CV_32F);
    Mat responseIndices = Mat(0, 0, CV_32S);
    for (string response: responses) {
        responseIndices.push_back(getIndexOfFruit(response));
    }
    Ptr<ml::SVM> svm = createAndTrainSvm(reducedFeatures, responseIndices);

//    predictTrainingData(reducedFeatures, responseIndices, svm);

    while (true) {
        string filename;
        cout << "Dateiname eingeben: " << endl;
        getline(cin, filename);
        if (filename == "quit") {
            break;
        }
        Mat rgbImage = imread(filename);
        if (!rgbImage.data) {
            printf("No image data \n");
        } else {
            vector<double> extractedFeatures = extractFeatures(rgbImage);
            Mat testImage((int) extractedFeatures.size(), 1, CV_64F, extractedFeatures.data());
            testImage = pca->project(testImage.t());
            testImage.convertTo(testImage, CV_32F);
            cout << svm->predict(testImage.t()) << endl;
        }
    }
    return 0;
}