//
// Created by Long Bui on 19.04.17.
//

#include "Quadtree.h"

const int meanThreshold = 5;
const int minRegionSize = 4;

bool checkHomogeneityRegion(Mat image, Scalar &mean);

bool checkHomogeneityNeighbours(Mat image1, Mat image2, Scalar &mean);

Quadtree::Quadtree(Mat image) {
    this->root = shared_ptr<Node>(new Node(image));
}

Quadtree::~Quadtree() {
}

void Quadtree::splitAndMerge() {
    this->split();
    this->merge();
}

void Quadtree::split() {
    this->root->split();
}

void Quadtree::merge() {
    this->root->image = this->root->merge();
}

bool checkHomogeneityRegion(Mat image, Scalar &mean) {
    if (image.rows == 0 || image.cols == 0) {
        return true;
    }
    Scalar stddev;
    meanStdDev(image, mean, stddev);
    // Homogen, falls Standardabweichung < 5.8 oder Anzahl Pixel im Segment <= 25
    return (stddev[0] <= meanThreshold) || (image.rows * image.cols <= minRegionSize);
}

bool checkHomogeneityNeighbours(Mat image1, Mat image2, Scalar &mean) {
    Mat combinedRegion;
    if (image1.rows * image1.cols != image2.rows * image2.cols) {
        resize(image1, combinedRegion, image2.size(), 0, 0, CV_INTER_LINEAR);
        combinedRegion.push_back(image2);
    } else {
        combinedRegion.push_back(image1);
        combinedRegion.push_back(image2);
    }
    Scalar stddev;
    meanStdDev(combinedRegion, mean, stddev);

    // Homogen, falls Standardabweichung < 5.8 oder Anzahl Pixel im Segment <= 25
    return (stddev[0] <= meanThreshold) || (combinedRegion.rows * combinedRegion.cols <= minRegionSize);
}

void Quadtree::Node::split() {
    Scalar mean;
    if (!checkHomogeneityRegion(image, mean)) {
        upperLeft = shared_ptr<Node>(new Node(Mat(image,
                                                  Range(0, image.rows / 2),
                                                  Range(0, image.cols / 2))));
        lowerLeft = shared_ptr<Node>(new Node(Mat(image,
                                                  Range(image.rows / 2, image.rows),
                                                  Range(0, image.cols / 2))));
        lowerRight = shared_ptr<Node>(new Node(Mat(image,
                                                   Range(image.rows / 2, image.rows),
                                                   Range(image.cols / 2, image.cols))));
        upperRight = shared_ptr<Node>(new Node(Mat(image,
                                                   Range(0, image.rows / 2),
                                                   Range(image.cols / 2, image.cols))));
        upperLeft->split();
        lowerLeft->split();
        lowerRight->split();
        upperRight->split();
    } else {
        image.setTo(mean[0]);
    }
}

Mat Quadtree::Node::merge() {
    if (upperLeft == 0) {
        return image;
    } else {
        Mat mergedUpperLeft = upperLeft->merge();
        Mat mergedLowerLeft = lowerLeft->merge();
        Mat mergedLowerRight = lowerRight->merge();
        Mat mergedUpperRight = upperRight->merge();
        Scalar mean;
        if (checkHomogeneityNeighbours(mergedUpperLeft, mergedLowerLeft, mean)) {
            mergedUpperLeft.setTo(mean[0]);
            mergedLowerLeft.setTo(mean[0]);
        }
        if (checkHomogeneityNeighbours(mergedLowerLeft, mergedLowerRight, mean)) {
            mergedLowerLeft.setTo(mean[0]);
            mergedLowerRight.setTo(mean[0]);
        }
        if (checkHomogeneityNeighbours(mergedLowerRight, mergedUpperRight, mean)) {
            mergedLowerRight.setTo(mean[0]);
            mergedUpperRight.setTo(mean[0]);
        }
        vconcat(mergedUpperLeft, mergedLowerLeft, mergedUpperLeft);
        vconcat(mergedUpperRight, mergedLowerRight, mergedUpperRight);
        hconcat(mergedUpperLeft, mergedUpperRight, mergedUpperLeft);
        return mergedUpperLeft;
    }
}
