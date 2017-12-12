cmake_minimum_required(VERSION 2.8)
project(ch6)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_DIRS})

add_executable(curve_fitting main.cpp)
target_link_libraries(curve_fitting ${CERES_LIBRARIES} ${OpenCV_LIBS})

add_executable(g2o_op g2o_op.cpp)

include_directories("/usr/include/eigen3")
target_link_libraries(g2o_op ${OpenCV_LIBS} g2o_core g2o_stuff)