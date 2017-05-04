#include <iostream>
#include <opencv2/opencv.hpp>
#include <memory>
#include "segmentation/Quadtree.h"
#include "feature-extraction/Color.h"

using namespace cv;
using namespace std;

vector<String, allocator<String>> getAllFileNames();

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

vector<String> getAllFileNames() {
    String fruits[] = {
            "acerolas", "apples", "apricots", "avocados", "bananas", "blackberries", "blueberries",
            "cantaloupes", "cherries", "coconuts", "figs", "grapefruits", "grapes", "guava", "kiwifruit",
            "lemons", "limes", "mangos", "olives", "oranges", "passionfruit", "peaches", "pears", "pineapples",
            "plums", "pomegranates", "raspberries", "strawberries", "tomatoes", "watermelons"};

    vector<String> filenames;
    String folder = "FIDS30/";

    for (String fruit: fruits) {
        vector<String> filesForFruit;
        glob(folder + fruit, filesForFruit);
        filenames.insert(filenames.end(), filesForFruit.begin(), filesForFruit.end());
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
    namedWindow("reduced", CV_WINDOW_AUTOSIZE);
    namedWindow("normal", CV_WINDOW_AUTOSIZE);

    vector<String> filenames = getAllFileNames();

    for (String file : filenames) {
        Mat rgbImage;

        rgbImage = imread(file);

        if (!rgbImage.data) {
            cout << file << endl;
            printf("No image data \n");
        } else {
//            Mat grayImage;
//            cvtColor(rgbImage, grayImage, COLOR_RGB2GRAY);
//
//            shared_ptr<Quadtree> root(new Quadtree(grayImage));
//            root->splitAndMerge();
//
//            Mat thresholdImage;
//            fillHolesInThreshold(grayImage, thresholdImage);
//
//            segmentImage(rgbImage, thresholdImage);
//            resize(rgbImage, rgbImage, Size(256, 256));
//            imshow("Display frame", rgbImage);
            imshow("normal", rgbImage);
            extractColorHistogram(rgbImage);
            imshow("reduced", rgbImage);
            waitKey(0);
        }
    }
    return 0;
}

