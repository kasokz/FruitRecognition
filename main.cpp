#include <iostream>
#include <opencv2/opencv.hpp>
#include "segmentation/Quadtree.h"

using namespace cv;

void splitAndMerge(shared_ptr<Quadtree> tree, double thresholdValue);

bool checkHomogeneityRegion(Mat image);

void segmentImage(Mat &rgbImage, const Mat &thresholdImage) {
    for (int row = 0; row < rgbImage.rows; ++row) {
        for (int col = 0; col < rgbImage.cols; ++col) {
            if (thresholdImage.at<uchar>(row, col)) {
                Vec3b *pixel = rgbImage.ptr<Vec3b>(row, col);
                (*pixel)[0] = 255;
                (*pixel)[1] = 255;
                (*pixel)[2] = 255;
            }
        }
    }
}

int main(int argc, char **argv) {

    if (argc != 2) {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat rgbImage;
    Mat grayImage;
    Mat splitAndMergeImage;

    rgbImage = imread(argv[1], 1);

    if (!rgbImage.data) {
        printf("No image data \n");
        return -1;
    }

    cvtColor(rgbImage, grayImage, COLOR_RGB2GRAY);
    cvtColor(rgbImage, splitAndMergeImage, COLOR_RGB2GRAY);

    shared_ptr<Quadtree> root(new Quadtree(splitAndMergeImage));
    root->splitAndMerge();

    Mat thresholdImage;
    double threshholdValue = threshold(splitAndMergeImage, thresholdImage, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);

    namedWindow("original", WINDOW_AUTOSIZE);
    imshow("original", rgbImage);

    namedWindow("Split and Merge", WINDOW_AUTOSIZE);
    imshow("Split and Merge", root->getImage());

    namedWindow("gray", WINDOW_AUTOSIZE);
    imshow("gray", grayImage);

    namedWindow("otsu", WINDOW_AUTOSIZE);
    imshow("otsu", thresholdImage);

    namedWindow("segmented", WINDOW_AUTOSIZE);
    segmentImage(rgbImage, thresholdImage);
    imshow("segmented", rgbImage);

    waitKey(0);

    return 0;
}