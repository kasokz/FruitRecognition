cmake_minimum_required(VERSION 3.7)
project(FruitRecognition)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp segmentation/Quadtree.cpp segmentation/Quadtree.h feature-extraction/Color.cpp feature-extraction/Texture.cpp feature-extraction/Shape.cpp feature-extraction/Color.h feature-extraction/Shape.h feature-extraction/Texture.h feature-extraction/PrincipalComponentAnalysis.cpp feature-extraction/PrincipalComponentAnalysis.h)
add_executable(FruitRecognition ${SOURCE_FILES})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
target_link_libraries(FruitRecognition ${OpenCV_LIBS})