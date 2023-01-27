cmake_minimum_required(VERSION 3.3)

find_package(Pangolin REQUIRED)
find_package(OpenCV REQUIRED)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(distortionTest distortion_test.cpp)
target_link_libraries(distortionTest ${OpenCV_LIBS} ${Pangolin_LIBRARIES} ${PCL_LIBRARIES})
