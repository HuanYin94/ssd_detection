cmake_minimum_required(VERSION 2.8.3)

project(ssd_detection)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  cmake_modules
  nodelet
  libpointmatcher_ros
  cv_bridge
  message_filters
)

find_package(Eigen REQUIRED)
find_package(libpointmatcher REQUIRED)
find_package(Caffe)
#find_package(OpenCV REQUIRED)

add_definitions(-std=c++11 -O2 -g -DNDEBUG)
add_definitions(${Caffe_DEFINITIONS}) 

catkin_package (
#	CATKIN_DEPENDS roscpp
)

include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIRS}
  ${libpointmatcher_INCLUDE_DIRS}
  ${OPENCV_INCLUDE_DIRS}
)
message("opencvvvvv: " ${OPENCV_INCLUDE_DIRS})
include_directories(${Caffe_INCLUDE_DIRS})
#include_directories(/usr/include/opencv2)

#link_libraries("/home/yh/opencv/build/lib/libopencv_imgcodecs.so")

add_executable(ssd_detection src/ssd_detection.cpp)
target_link_libraries(ssd_detection 
${catkin_LIBRARIES} 
${Caffe_LIBRARIES}
${libpointmatcher_LIBRARIES}
${OpenCV_LIBS}
#/opt/ros/indigo/lib/libopencv_videoio3.so 
#/home/yh/opencv/build/lib/libopencv_imgcodecs.so
)
