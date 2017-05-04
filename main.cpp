#include <iostream>
#include <opencv2/opencv.hpp>
#include <memory>
#include "segmentation/Quadtree.h"
#include "feature-extraction/Color.h"
#include "feature-extraction/Texture.h"
#include "feature-extraction/PrincipalComponentAnalysis.h"
#include <eigen3/Eigen/Core>

using namespace cv;
using namespace std;

map<String, vector<String>> getAllFileNames();

String fruits[] = {
        "acerolas", "apples", "apricots", "avocados", "bananas", "blackberries", "blueberries",
        "cantaloupes", "cherries", "coconuts", "figs", "grapefruits", "grapes", "guava", "kiwifruit",
        "lemons", "limes", "mangos", "olives", "oranges", "passionfruit", "peaches", "pears", "pineapples",
        "plums", "pomegranates", "raspberries", "strawberries", "tomatoes", "watermelons"};


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

int main(int argc, char **argv) {
    namedWindow("Display frame", CV_WINDOW_AUTOSIZE);

    map<String, vector<String>> filenames = getAllFileNames();

    for (int i = 0; i < fruits->length(); i++) {
        for (String fruitFile: filenames[fruits[i]]) {
            Mat rgbImage;

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
//                imshow("Display frame", rgbImage);
                cout << rgbImage.isContinuous() << endl;
                vector<double> extractedFeatures = extractColorHistogram(rgbImage);
                performPCA(extractedFeatures, 14);
                waitKey(0);
            }
        }
    }
    return 0;
}

