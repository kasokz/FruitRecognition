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
        "apples", "apricots", "avocados", "bananas", "blackberries", "blueberries",
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
    map<String, vector<String>> filenames = getAllFileNames();

    for (int i = 0; i < fruits->length(); i++) {
        shared_ptr<PrincipalComponentAnalysis> pca(new PrincipalComponentAnalysis());
//        vector<double> a = {2.5, 2.4};
//        vector<double> b = {0.5, 0.7};
//        vector<double> c = {2.2, 2.9};
//        vector<double> d = {1.9, 2.2};
//        vector<double> e = {3.1, 3.0};
//        vector<double> f = {2.3, 2.7};
//        vector<double> g = {2, 1.6};
//        vector<double> h = {1, 1.1};
//        vector<double> k = {1.5, 1.6};
//        vector<double> j = {1.1, 0.9};
//        pca->addFruitData(a);
//        pca->addFruitData(b);
//        pca->addFruitData(c);
//        pca->addFruitData(d);
//        pca->addFruitData(e);
//        pca->addFruitData(f);
//        pca->addFruitData(g);
//        pca->addFruitData(h);
//        pca->addFruitData(k);
//        pca->addFruitData(j);


        for (String fruitFile: filenames[fruits[i]]) {
            Mat rgbImage;

            rgbImage = imread(fruitFile);

            if (!rgbImage.data) {
                cout << fruitFile << endl;
                printf("No image data \n");
            } else {
                int px = 300;
                Size size(px, px);
                Mat grayImage, outputImage;
                rgbImage.copyTo(outputImage);
                cvtColor(rgbImage, grayImage, COLOR_RGB2GRAY);
                resize(outputImage, outputImage, size);
                imshow("1. Input RGB Image", outputImage);
                moveWindow("1. Input RGB Image", 0, 0);


                grayImage.copyTo(outputImage);
                resize(outputImage, outputImage, size);
                imshow("2. normal Image", outputImage);
                moveWindow("2. normal Image", px + 10, 0);

                shared_ptr<Quadtree> root(new Quadtree(grayImage));
                root->splitAndMerge();

                grayImage.copyTo(outputImage);
                resize(outputImage, outputImage, size);
                imshow("3. Gray Image after Split and Merge", outputImage);
                moveWindow("3. Gray Image after Split and Merge", 2 * (px + 10), 0);

                Mat thresholdImage;
                fillHolesInThreshold(grayImage, thresholdImage);

                thresholdImage.copyTo(outputImage);
                resize(outputImage, outputImage, size);
                imshow("4. Otsu's treshold with hole filling", outputImage);
                moveWindow("4. Otsu's treshold with hole filling", 0, px + 50);

                segmentImage(rgbImage, thresholdImage);

                resize(rgbImage, rgbImage, size);
                imshow("Final image", rgbImage);
                moveWindow("Final image", px + 10, px + 50);

                vector<double> extractedFeatures = extractColorHistogram(rgbImage);
                vector<double> textures = unser(grayImage);
                extractedFeatures.insert(extractedFeatures.end(), textures.begin(), textures.end());
                pca->addFruitData(extractedFeatures);

//                unserTest(grayImage);
//                waitKey(0);
            }
        }
        cout << pca->performPCA(14) << endl;
        waitKey(0);
    }
    return 0;
}

