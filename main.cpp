#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include "segmentation/Quadtree.h"
#include "feature-extraction/Color.h"
#include "feature-extraction/Texture.h"
#include "feature-extraction/PrincipalComponentAnalysis.h"
#include "feature-extraction/Shape.h"

using namespace cv;
using namespace std;

struct ImageWithClass {
    Mat rgbImage;
    String fruit;

    ImageWithClass(const Mat &rgbImage, const String &fruit) : rgbImage(rgbImage), fruit(fruit) {}
};

map<String, vector<String>> getAllFileNames();

vector<vector<double>> readCsvDataset(vector<string> &responses);

Mat convertToMat(vector<vector<double>> data);

void predictTrainingData(const Mat &reducedFeatures, Mat &responseIndices, const Ptr<ml::SVM> &svm);

vector<double> extractFeatures(Mat &rgbImage);

void segmentImage(Mat &rgbImage, const Mat &thresholdImage);

Ptr<ml::SVM> createAndTrainSvm(Mat features, Mat responses);

void writeFeaturesToFile(const vector<vector<double>> &featuresThread1, const vector<String> &fruitsThread1,
                         ofstream &csvFile);

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
    String folder = "fruit_dataset/";
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

void threadFunction(vector<ImageWithClass> imagesWithClass, vector<vector<double>> &features,
                    vector<String> &fruits) {
    for (ImageWithClass imageWithClass: imagesWithClass) {
        vector<double> extractedFeatures = extractFeatures(imageWithClass.rgbImage);
        features.push_back(extractedFeatures);
        fruits.push_back(imageWithClass.fruit);
    }
}

void createDatasetAsCsv() {

    map<String, vector<String>> filenames = getAllFileNames();
    vector<vector<ImageWithClass>> threadDatasets;
    threadDatasets.push_back(vector<ImageWithClass>());
    threadDatasets.push_back(vector<ImageWithClass>());
    threadDatasets.push_back(vector<ImageWithClass>());
    threadDatasets.push_back(vector<ImageWithClass>());
    int indexOfDataset = 0;
    for (String fruit: fruits) {
        for (String fruitFile: filenames[fruit]) {
            Mat rgbImage;
            rgbImage = imread(fruitFile);
            if (!rgbImage.data) {
                cout << fruitFile << endl;
                printf("No image data \n");
            } else {
                threadDatasets[indexOfDataset % 4].push_back(ImageWithClass(rgbImage, fruit));
                indexOfDataset++;
            }
        }
    }
    vector<vector<double>> featuresThread1;
    vector<vector<double>> featuresThread2;
    vector<vector<double>> featuresThread3;
    vector<vector<double>> featuresThread4;
    vector<String> fruitsThread1;
    vector<String> fruitsThread2;
    vector<String> fruitsThread3;
    vector<String> fruitsThread4;
    thread thread1(threadFunction, threadDatasets[0], ref(featuresThread1), ref(fruitsThread1));
    thread thread2(threadFunction, threadDatasets[1], ref(featuresThread2), ref(fruitsThread2));
    thread thread3(threadFunction, threadDatasets[2], ref(featuresThread3), ref(fruitsThread3));
    thread thread4(threadFunction, threadDatasets[3], ref(featuresThread4), ref(fruitsThread4));
    thread1.join();
    thread2.join();
    thread3.join();
    thread4.join();

    const int colors = 64;
    const int textures = 8;
    const int shapes = 0;

    ofstream csvFile;
    csvFile.open("fruit_features.csv");
    for (int i = 0; i < colors; i++) {
        csvFile << "color" << i << ",";
    }
    for (int i = 0; i < textures; i++) {
        csvFile << "unser" << i << ",";
    }
    for (int i = 0; i < shapes; i++) {
        csvFile << "shape" << i << ",";
    }
    csvFile << "class" << endl;

    writeFeaturesToFile(featuresThread1, fruitsThread1, csvFile);
    writeFeaturesToFile(featuresThread2, fruitsThread2, csvFile);
    writeFeaturesToFile(featuresThread3, fruitsThread3, csvFile);
    writeFeaturesToFile(featuresThread4, fruitsThread4, csvFile);

    csvFile.close();
}

void writeFeaturesToFile(const vector<vector<double>> &featuresThread, const vector<String> &fruitsThread,
                         ofstream &csvFile) {
    for (int i = 0; i < featuresThread.size(); i++) {
        vector<double> extractedFeatures = featuresThread[i];
        Mat features = Mat(1, (int) extractedFeatures.size(), CV_64F, extractedFeatures.data());
        csvFile << format(features, Formatter::FMT_CSV);
        csvFile << "," << fruitsThread[i] << endl;
    }
}

vector<double> extractFeatures(Mat &rgbImage) {
    Mat grayImage;
    cvtColor(rgbImage, grayImage, COLOR_RGB2GRAY);

    shared_ptr<Quadtree> root(new Quadtree(grayImage));
    root->splitAndMerge();

    Mat thresholdImage;
    fillHolesInThreshold(grayImage, thresholdImage);

    segmentImage(rgbImage, thresholdImage);

    vector<double> extractedFeatures = extractColorHistogram(rgbImage);
    vector<double> textures = unser(grayImage);
//    vector<double> shapes = shape(grayImage);
    extractedFeatures.insert(extractedFeatures.end(), textures.begin(), textures.end());
//    extractedFeatures.insert(extractedFeatures.end(), shapes.begin(), shapes.end());
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
            if (count == 82) {
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

void predictTrainingData(const Mat &reducedFeatures, Mat &responseIndices, const Ptr<ml::SVM> &svm) {
    Mat result;
    svm->predict(reducedFeatures.t(), result);

    int correctPredictions = 0;
    int sumPredictions = 0;
    for (int i = 0; i < responseIndices.rows; i++) {
//        cout << "Predicted: " << fruits[(int) result.at<float>(i)] << ", Actual: " << fruits[responseIndices.at<int>(i)]
//             << endl;
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

void runApplication() {
    vector<string, allocator<string>> responses;
    vector<vector<double>> all_data = readCsvDataset(responses);
    Mat dataAsMat = convertToMat(all_data);

    shared_ptr<PrincipalComponentAnalysis> pca(new PrincipalComponentAnalysis());
    for (vector<double> a: all_data) {
        pca->addFruitData(a);
    }

    for (int i = 1; i <= 84; i++) {
        pca->fit(i);

        Mat reducedFeatures = pca->project(dataAsMat);

        reducedFeatures.convertTo(reducedFeatures, CV_32F);
        Mat responseIndices = Mat(0, 0, CV_32S);
        for (string response: responses) {
            responseIndices.push_back(getIndexOfFruit(response));
        }
        Ptr<ml::SVM> svm = createAndTrainSvm(reducedFeatures, responseIndices);

        predictTrainingData(reducedFeatures, responseIndices, svm);
    }
//    while (true) {
//        string filename;
//        cout << "Dateiname eingeben: " << endl;
//        getline(cin, filename);
//        if (filename == "quit") {
//            break;
//        }
//        Mat rgbImage = imread(filename);
//        if (!rgbImage.data) {
//            printf("No image data \n");
//        } else {
//            vector<double> extractedFeatures = extractFeatures(rgbImage);
//            Mat testImage((int) extractedFeatures.size(), 1, CV_64F, extractedFeatures.data());
//            testImage = pca->project(testImage.t());
//            testImage.convertTo(testImage, CV_32F);
//            cout << svm->predict(testImage.t()) << endl;
//
//
//        }
//    }
}

int main(int argc, char **argv) {
//    createDatasetAsCsv();
    runApplication();
    return 0;
}