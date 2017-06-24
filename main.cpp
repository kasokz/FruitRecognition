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

string decorateNumber(char character, int number, char character2);

const int numOfColorFeatures = 64;
const int numOfTextureFeatures = 8;
const int numOfShapeFeatures = 7;

void printDataToCsv(const Mat &data, const Mat &trainingResponseAsIndex);

void performTest(int componentCount,
                 const Mat &trainingDataAsMat, const Mat &testDataAsMat,
                 const Mat &trainingResponsesAsIndex, const Mat &testResponsesAsIndex);

map<String, vector<String>> getAllFileNames(String folder);

void readCsvDataset(vector<vector<double>> &trainingDataset, vector<vector<double>> &testDataset,
                    vector<string> &trainingResponses, vector<string> &testResponses);

Mat convertToMat(vector<vector<double>> data);

void startPredictionWithData(const Mat &reducedFeatures, const Mat &responseIndices, const Ptr<ml::SVM> &svm);

vector<double> extractFeatures(Mat &rgbImage);

void segmentImage(Mat &rgbImage, const Mat &thresholdImage);

Ptr<ml::SVM> createAndTrainSvm(Mat features, Mat responses);

void writeFeaturesToFile(const vector<vector<double>> &featuresThread1, const vector<String> &fruitsThread1,
                         ofstream &csvFile);

void startCLI(shared_ptr<PrincipalComponentAnalysis> &pca, const Ptr<ml::SVM> &svm);

void printConfusionMat(const Mat &confusionMat);

String fruits[] = {
        "red_apples",
//        "apricots",
        "avocados",
        "bananas",
//        "blackberries",
//        "blueberries",
//        "cantaloupes",
//        "cherries",
//        "coconuts",
//        "grapes",
        "green_apples",
        "green_plantains",
//        "lemons",
//        "limes",
//        "mangos",
//        "oranges",
//        "passionfruit",
//        "peaches",
//        "pears",
//        "pineapples",
//        "plums",
        "raspberries",
        "strawberries",
        "tomatoes",
        "watermelons"
};

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

map<String, vector<String>> getAllFileNames(String folder) {
    map<String, vector<String>> filenames;
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
    map<String, vector<String>> filenames = getAllFileNames("fruit_dataset/");
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

    ofstream csvFile;
    csvFile.open("fruit_features.csv");
    for (int i = 0; i < numOfColorFeatures; i++) {
        csvFile << "color" << i << ",";
    }
    for (int i = 0; i < numOfTextureFeatures; i++) {
        csvFile << "unser" << i << ",";
    }
    for (int i = 0; i < numOfShapeFeatures; i++) {
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
    cvtColor(rgbImage, grayImage, COLOR_RGB2GRAY);

    vector<double> extractedFeatures;
    vector<double> colors = extractColorHistogram(rgbImage);
    vector<double> textures = unser(grayImage);
    vector<double> shapes = shape(grayImage);
    for (int i = 0; i < numOfColorFeatures; i++) {
        extractedFeatures.push_back(colors[i]);
    }
    for (int i = 0; i < numOfTextureFeatures; i++) {
        extractedFeatures.push_back(textures[i]);
    }
    for (int i = 0; i < numOfShapeFeatures; i++) {
        extractedFeatures.push_back(shapes[i]);
    }
    return extractedFeatures;
}

void readCsvDataset(vector<vector<double>> &trainingDataset, vector<vector<double>> &testDataset,
                    vector<string> &trainingResponses, vector<string> &testResponses) {
    ifstream inputfile("fruit_features.csv");
    string current_line;
    int lineCounter = 0;
    while (getline(inputfile, current_line)) {
        if (lineCounter++ != 0) {
            vector<double> values;
            string response;
            stringstream temp(current_line);
            string currentToken;
            int count = 0;
            while (getline(temp, currentToken, ',')) {
                if (count++ == numOfColorFeatures + numOfTextureFeatures + numOfShapeFeatures) {
                    response = currentToken.c_str();
                } else {
                    values.push_back(atof(currentToken.c_str()));
                }
            }
            if (lineCounter % 5 == 0) {
                testDataset.push_back(values);
                testResponses.push_back(response);
            } else {
                trainingDataset.push_back(values);
                trainingResponses.push_back(response);
            }
        }
    }
}

void startPredictionWithData(const Mat &reducedFeatures, const Mat &responseIndices, const Ptr<ml::SVM> &svm) {
    Mat result;
    Mat confusionMat = Mat::zeros(sizeof(fruits) / sizeof(*fruits), sizeof(fruits) / sizeof(*fruits), CV_32S);
    svm->predict(reducedFeatures.t(), result);
    int correctPredictions = 0;
    int sumPredictions = 0;
    for (int i = 0; i < responseIndices.rows; i++) {
//        cout << "Predicted: " << fruits[(int) result.at<float>(i)] << ", Actual: " << fruits[responseIndices.at<int>(i)]
//             << endl;
        confusionMat.at<int>((int) result.at<float>(i), responseIndices.at<int>(i))++;
        sumPredictions++;
        if (result.at<float>(i) == responseIndices.at<int>(i)) {
            correctPredictions++;
        }
    }
    cout << "Correct Predictions: " << correctPredictions << "(" << setprecision(4)
         << ((double) correctPredictions / sumPredictions) * 100 << "%)" << endl;
//    printConfusionMat(confusionMat);
}

string decorateNumber(char character1, int number, char character2) {
    std::stringstream stringStream;
    stringStream << character1 << number << character2;
    return stringStream.str();
}

void printConfusionMat(const Mat &confusionMat) {
    const char separator = ' ';
    const int nameWidth = 16;
    const int rowNumberWidth = 5;
    const int colNumberWidth = 6;


    cout << "Confusion Matrix:" << endl;
    cout << left << setw(nameWidth + rowNumberWidth) << setfill(separator) << "";
    for (int i = 0; i < sizeof(fruits) / sizeof(*fruits); i++) {
        cout << left << setw(colNumberWidth) << setfill(separator) << decorateNumber('[', i + 1, ']');
    }
    cout << endl;
    for (int i = 0; i < confusionMat.rows; i++) {
        cout << left << setw(rowNumberWidth) << setfill(separator) << decorateNumber('[', i + 1, ']');
        cout << left << setw(nameWidth) << setfill(separator) << fruits[i];
        for (int j = 0; j < confusionMat.cols; j++) {
            string toPrint;
            if (i == j) {
                toPrint = decorateNumber('(', confusionMat.at<int>(i, j), ')');

            } else {
                toPrint = decorateNumber(' ', confusionMat.at<int>(i, j), ' ');
            }
            cout << left << setw(colNumberWidth) << setfill(separator)
                 << toPrint;
        }
        cout << endl;
    }
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

void startCLI(shared_ptr<PrincipalComponentAnalysis> &pca, const Ptr<ml::SVM> &svm) {
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
}

void printDataToCsv(const Mat &data, const Mat &trainingResponseAsIndex) {
    ofstream csvFile;
    csvFile.open("pca_features.csv");
    csvFile << "compo1,compo2,compo3,fruit" << endl;
    Mat transposed = data.t();
    for (int i = 0; i < transposed.rows; i++) {
        csvFile << format(transposed.row(i), Formatter::FMT_CSV) << "," << trainingResponseAsIndex.at<int>(i) << endl;
    }
}

void performTest(int componentCount,
                 const Mat &trainingDataAsMat, const Mat &testDataAsMat,
                 const Mat &trainingResponsesAsIndex, const Mat &testResponsesAsIndex) {
    shared_ptr<PrincipalComponentAnalysis> pca(new PrincipalComponentAnalysis());
    pca->fit(trainingDataAsMat, componentCount);

    Mat reducedTrainingData = pca->project(trainingDataAsMat);

//    printDataToCsv(reducedTrainingData, trainingResponsesAsIndex);

    reducedTrainingData.convertTo(reducedTrainingData, CV_32F);
    Ptr<ml::SVM> svm = createAndTrainSvm(reducedTrainingData, trainingResponsesAsIndex);

    Mat reducedTestData = pca->project(testDataAsMat);
    reducedTestData.convertTo(reducedTestData, CV_32F);

    cout << "Feature Components: " << componentCount << endl;
    startPredictionWithData(reducedTestData, testResponsesAsIndex, svm);
}

void runApplication() {
    vector<vector<double>> trainingDataset, testDataset;
    vector<string> trainingResponses, testResponses;
    readCsvDataset(trainingDataset, testDataset, trainingResponses,
                   testResponses);
    Mat trainingDataAsMat = convertToMat(trainingDataset);
    Mat testDataAsMat = convertToMat(testDataset);
    Mat trainingResponsesAsIndex = Mat(0, 0, CV_32S);
    for (string response: trainingResponses) {
        trainingResponsesAsIndex.push_back(getIndexOfFruit(response));
    }
    Mat testResponsesAsIndex = Mat(0, 0, CV_32S);
    for (string response: testResponses) {
        testResponsesAsIndex.push_back(getIndexOfFruit(response));
    }
    thread threads[4];

    for (int i = 1; i <= numOfColorFeatures + numOfTextureFeatures + numOfShapeFeatures; i++) {
        threads[(i - 1) % (sizeof(threads) / sizeof(threads[0]))] = thread(performTest, i,
                                                                           ref(trainingDataAsMat), ref(testDataAsMat),
                                                                           ref(trainingResponsesAsIndex),
                                                                           ref(testResponsesAsIndex));
        if (i != 0 && i % (sizeof(threads) / sizeof(threads[0])) == 0) {
            for (int threadIndex = 0; threadIndex < (sizeof(threads) / sizeof(threads[0])); threadIndex++) {
                threads[threadIndex].join();
            }
        }
    }
//    performTest(3, trainingDataAsMat, testDataAsMat, trainingResponsesAsIndex, testResponsesAsIndex);
//    startCLI(pca, svm);
}

int px, delta;
int posX(int x){
    return x * (px + delta);
}
int posY(int y){
    return y * (px + delta) + (y == 1 ? 30 : 0);
}
void show(){
    map<String, vector<String>> filenames = getAllFileNames("fruit_dataset_big/");

    for (int i = 0; i < fruits->length(); i++) {
        shared_ptr<PrincipalComponentAnalysis> pca(new PrincipalComponentAnalysis());
        for (int j = 0; j < filenames[fruits[i]].size(); ++j) {
            String fruitFile = filenames[fruits[i]][j];
            Mat rgbImage;
            rgbImage = imread(fruitFile);

            if (!rgbImage.data) {
                cout << fruitFile << endl;
                printf("No image data \n");
            } else {
                px = 450;
                delta = 10;
                Size size(px, px);
                Mat grayImage, outputImage;
                rgbImage.copyTo(outputImage);
                resize(outputImage, outputImage, size);
                imshow("1. Input Image", outputImage);
                moveWindow("1. Input Image", posX(0), posY(0));

                cvtColor(rgbImage, grayImage, COLOR_RGB2GRAY);
                shared_ptr<Quadtree> root(new Quadtree(grayImage));
                root->splitAndMerge();

                grayImage.copyTo(outputImage);
                resize(outputImage, outputImage, size);
                imshow("2. Grayscale Image after Split and Merge", outputImage);
                moveWindow("2. Grayscale Image after Split and Merge", posX(1), posY(0));

                Mat thresholdImage;
                fillHolesInThreshold(grayImage, thresholdImage);

                thresholdImage.copyTo(outputImage);
                resize(outputImage, outputImage, size);
                imshow("3. Otsu's treshold with hole filling", outputImage);
                moveWindow("3. Otsu's treshold with hole filling", posX(2), posY(0));

                segmentImage(rgbImage, thresholdImage);

                resize(rgbImage, rgbImage, size);
                imshow("4. Segmented Image", rgbImage);
                moveWindow("4. Segmented Image", posX(3), posY(0));

                extractColorHistogram(rgbImage);
                moveWindow("Reduced colors", posX(0) + px/2, posY(1));
                Mat gray2;
                cvtColor(rgbImage, gray2, COLOR_RGB2GRAY);
                shape(gray2);
                moveWindow("Convex Hull", posX(1) + px/2, posY(1));
                moveWindow("Ellipse", posX(2) + px/2, posY(1));
                if (waitKey(0) == 'a' && j != 0)
                    j -= 2;
            }
        }
    }
}

int main(int argc, char **argv) {
    createDatasetAsCsv();
    runApplication();
    //show();
    return 0;
}