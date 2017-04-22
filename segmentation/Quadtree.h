//
// Created by Long Bui on 19.04.17.
//
#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>

#ifndef FRUITRECOGNITION_QUADTREE_H
#define FRUITRECOGNITION_QUADTREE_H

using namespace std;
using namespace cv;

class Quadtree {
private:
    struct Node {
        Mat image;
        shared_ptr<Node> upperLeft;
        shared_ptr<Node> lowerLeft;
        shared_ptr<Node> lowerRight;
        shared_ptr<Node> upperRight;

        Node(Mat image) :
                image(image), upperLeft(0), lowerLeft(0), lowerRight(0), upperRight(0) {}

        void split();

        Mat merge();
    };

    shared_ptr<Node> root;

public:
    Quadtree(Mat image);

    void splitAndMerge();

    void split();

    void merge();

    virtual ~Quadtree();
};


#endif //FRUITRECOGNITION_QUADTREE_H
