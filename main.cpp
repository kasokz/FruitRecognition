#include <iostream>
#include <opencv2/opencv.hpp>
#include <memory>
#include "segmentation/Quadtree.h"
#include "feature-extraction/Color.h"
#include "feature-extraction/Texture.h"
#include "feature-extraction/PrincipalComponentAnalysis.h"

using namespace cv;
using namespace std;

map<String, vector<String>> getAllFileNames();

String fruits[] = {
        "apples", "apricots", "avocados", "bananas", "blackberries", "blueberries", "cantaloupes", "cherries",
        "coconuts", "figs", "grapefruits", "grapes", "guava", "kiwifruit", "lemons", "limes", "mangos", "olives",
        "oranges", "passionfruit", "peaches", "pears", "pineapples", "plums", "pomegranates", "raspberries",
        "strawberries", "tomatoes", "watermelons"};


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
    map<String, vector<String>> filenames = getAllFileNames();
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

    for (String fruit: fruits) {
        for (String fruitFile: filenames[fruit]) {
            Mat rgbImage;
            cout << fruitFile << endl;
            rgbImage = imread(fruitFile);
            if (!rgbImage.data) {
                cout << fruitFile << endl;
                printf("No image data \n");
            } else {
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
                Mat features = Mat(1, (int) extractedFeatures.size(), CV_64F);
                memcpy(features.data, extractedFeatures.data(), extractedFeatures.size() * sizeof(double));
                csvFile << cv::format(features, cv::Formatter::FMT_CSV);
                csvFile << "," << fruit << endl;
//                unserTest(grayImage);
            }
        }
    }
    csvFile.close();
}

int main(int argc, char **argv) {
//    createDatasetAsCsv();
    ifstream inputfile("fruit_features.csv");
    string current_line;
    vector<vector<double>> all_data;
    while (getline(inputfile, current_line)) {
        // Now inside each line we need to seperate the cols
        vector<double> values;
        stringstream temp(current_line);
        string single_value;
        while (getline(temp, single_value, ',')) {
            // convert the string element to a integer value
            values.push_back(atoi(single_value.c_str()));
        }
        // add the row to the complete data vector
        all_data.push_back(values);
    }

    for (vector<double> a : all_data) {
        for(double b : a) {
            cout << b << ",";
        }
        cout << endl;
    }

    shared_ptr<PrincipalComponentAnalysis> pca(new PrincipalComponentAnalysis());
    //    cout << pca->performPCA(14) << endl;

    return 0;
}

