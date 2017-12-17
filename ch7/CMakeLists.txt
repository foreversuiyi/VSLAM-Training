cmake_minimum_required(VERSION 2.8)
project(ch7)

set(CMAKE_BUILD_TYPE "debug")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.1 REQUIRED)
find_package(G2O REQUIRED)
find_package(CSparse REQUIRED)

include_directories(
	${OpenCV_INCLUDE_DIRS}
	${G2O_INCLUDE_DIRS}
	${CSPARSE_INCLUDE_DIR}
	"/usr/include/eigen3/"
)

add_executable(feature_extraction feature_extraction.cpp)
target_link_libraries(feature_extraction ${OpenCV_LIBS} )

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})

add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d 
${OpenCV_LIBS} ${CSPARSE_LIBRARY} g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)

add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d 
${OpenCV_LIBS} ${CSPARSE_LIBRARY} g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)