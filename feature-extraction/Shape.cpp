//
// Created by Long Bui on 25.04.17.
//

#include "Shape.h"

int area(const Mat &image) {
    int sum = 0;
    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
            if(image.at<uchar>(i,j) != 255)
                sum++;
    return sum;
}

int perimeterAlternative(const Mat &image)
{
    int perimeter = 0;
    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
            //if actual pixel belongs to object, not to background
            if(image.at<uchar>(i,j) != 255) {
                if(i == 0)
                    perimeter++;
                else if(image.at<uchar>(i-1,j) == 255)
                    perimeter++;
                if(i == image.rows-1)
                    perimeter++;
                else if(image.at<uchar>(i+1,j) == 255)
                    perimeter++;;
                if(j == 0)
                    perimeter++;
                else if(image.at<uchar>(i,j-1) == 255)
                    perimeter++;
                if(j == image.rows-1)
                    perimeter++;
                if(image.at<uchar>(i,j+1) == 255)
                    perimeter++;
            }
    return perimeter;
}

Mat contour(const Mat &image) {
    Mat buffer;
    image.copyTo(buffer);
    for (int i = 1; i < buffer.rows-1; ++i)
        for (int j = 1; j < buffer.cols-1; ++j)
            //if the actual pixel and its neightbors belong to object
            if(image.at<uchar>(i,j) != 255 &&
                    image.at<uchar>(i-1,j) != 255 &&
                    image.at<uchar>(i+1,j) != 255 &&
                    image.at<uchar>(i,j-1) != 255 &&
                    image.at<uchar>(i,j+1) != 255)
                buffer.at<uchar>(i,j) = 255;
    //imshow("Gray 2", buffer);moveWindow("Gray 2", 0, 0);
    return buffer;
}

void bresenham(Mat &image, int x1, int y1, int x2, int y2)
{
    int w = x2 - x1;
    int h = y2 - y1;
    int dx1 = 0, dy1 = 0, dx2 = 0, dy2 = 0;
    if (w<0) dx1 = -1; else if (w>0) dx1 = 1;
    if (h<0) dy1 = -1; else if (h>0) dy1 = 1;
    if (w<0) dx2 = -1; else if (w>0) dx2 = 1;
    int longest = abs(w);
    int shortest = abs(h);
    if (longest <= shortest){
        longest = abs(h);
        shortest = abs(w);
        if (h<0) dy2 = -1; else if (h>0) dy2 = 1;
        dx2 = 0;
    }
    int numerator = longest >> 1;
    for (int i = 0; i <= longest; i++) {
        image.at<uchar>(y1, x1) = 0;
        numerator += shortest;
        if (longest <= numerator){
            numerator -= longest;
            x1 += dx1;
            y1 += dy1;
        }
        else {
            x1 += dx2;
            y1 += dy2;
        }
    }
}

//number of pixels of the convex hull - Jarvis algorithm
int convexHull(const Mat &edgedImage) {
    vector<vector<int>> points; //all edge points of object
    //find the point with smallest ordinate (y-coordinate)
    int minY = INT_MAX;
    int startX = -1;
    int startY = -1;
    int startIndex = -1;
    for (int i = 0; i < edgedImage.rows; i++)
        for (int j = 0; j < edgedImage.cols; j++)
            if(edgedImage.at<uchar>(i,j) != 255) {
                points.push_back(vector<int> {i, j});
                if(i < minY) {
                    minY = i;
                    startX = j;
                    startY = i;
                    startIndex = points.size()-1;
                }
            }
    vector<vector<int>> hullPoints;
    int i = 0, endX = -1, endY = -1;
    do{
        hullPoints.push_back(vector<int> {startY, startX});
        endX = points[0][1];
        endY = points[0][0];
        for (int j = 1; j < points.size(); ++j)
            if((endX == startX && endY == startY) || ((points[j][1] - startX)*(endY - startY) - (points[j][0] - startY)*(endX - startX) < 0)){
                endX = points[j][1];
                endY = points[j][0];
            }
        startX = endX;
        startY = endY;
        i++;
    }while(!(endX == hullPoints[0][1] && endY == hullPoints[0][0]));
    Mat buffer = Mat::zeros(edgedImage.size(), CV_8UC1);
    for (int l = 0; l < buffer.rows; ++l)
        for (int j = 0; j < buffer.cols; ++j)
            buffer.at<uchar>(l,j) = 255;
    for (int k = 0; k < hullPoints.size(); ++k)
        bresenham(buffer, hullPoints[k][1], hullPoints[k][0], hullPoints[(k+1)%hullPoints.size()][1], hullPoints[(k+1)%hullPoints.size()][0]);
    return area(buffer);
}

vector<double> shape(const Mat &image) {
    vector<double> results;
    double areaValue = area(image);
    results.push_back(areaValue);
    Mat edges = contour(image);
    int perimeter = area(edges);
    results.push_back((double)perimeter);// perimeter of the object
    double hull = convexHull(edges);
    results.push_back(hull);
    results.push_back(areaValue/hull);
//    convexHull(edges);
//    imshow("Hull", edges);moveWindow("Hull", 500, 500);
    return results;
}