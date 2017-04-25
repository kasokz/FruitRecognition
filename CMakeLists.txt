cmake_minimum_required(VERSION 3.7)
project(FruitRecognition)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp segmentation/Quadtree.cpp segmentation/Quadtree.h feature-extraction/Color.cpp feature-extraction/Texture.cpp feature-extraction/Shape.cpp)
add_executable(FruitRecognition ${SOURCE_FILES})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(FruitRecognition ${OpenCV_LIBS})