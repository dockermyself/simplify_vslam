cmake_minimum_required( VERSION 2.8 )
project( simplify-vslam)

set( CMAKE_BUILD_TYPE Debug )
set( CMAKE_CXX_FLAGS "-std=c++11 -Wall -O2 -march=native" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package( G2O REQUIRED )
find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )
find_package( Cholmod )
find_package( Pangolin )
include_directories( ${EIGEN3_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR} ${Pangolin_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)

add_executable(slam slam.cpp)
target_link_libraries( slam ${OpenCV_LIBS} ${G2O_LIBRARY} ${CHOLMOD_LIBRARIES} ${Pangolin_LIBRARIES})
