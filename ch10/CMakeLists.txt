cmake_minimum_required(VERSION 2.8)
project(ch10)
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

Find_Package(G2O REQUIRED)
Find_Package(Eigen3 REQUIRED)
Find_Package(Cholmod REQUIRED)
Find_package(Ceres REQUIRED)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear 
    g2o_types_icp g2o_types_slam2d g2o_types_sba g2o_types_slam3d g2o_core g2o_interface 
    g2o_solver_csparse g2o_solver_structure_only g2o_csparse_extension g2o_opengl_helper g2o_solver_dense 
    g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse)

include_directories(${CERES_INCLUDE_DIRS}
					${EIGEN3_INCLUDE_DIR} 
					${CHOLMOD_INCLUDE_DIR} 
					${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(BA_LIB SHARED 
			src/BALProblem.cpp 
			src/command_args.cpp)

target_link_libraries(BA_LIB ${G2O_LIBS} ${CHOLMOD_LIBRARIES})

add_executable(g2o_ba test/g2o_ba.cpp)
add_executable(ceres_ba test/ceres_ba.cpp)

target_link_libraries(g2o_ba BA_LIB)
target_link_libraries(ceres_ba BA_LIB ${CERES_LIBRARIES} )