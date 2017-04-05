cmake_minimum_required(VERSION 2.8.3)
project(ssd_detection)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  cmake_modules
  nodelet
  libpointmatcher_ros
  image_transport
  cv_bridge
)

find_package(Eigen REQUIRED)
find_package(libpointmatcher REQUIRED)
find_package(Caffe)

add_definitions(-std=c++11 -O2 -g -DNDEBUG)
add_definitions(${Caffe_DEFINITIONS}) 

catkin_package (
#	CATKIN_DEPENDS roscpp
)

include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIRS}
  ${libpointmatcher_INCLUDE_DIRS}s
)
include_directories(${Caffe_INCLUDE_DIRS})

add_executable(ssd_detection src/ssd_detection.cpp)
target_link_libraries(ssd_detection 
${catkin_LIBRARIES} 
${Caffe_LIBRARIES}
${libpointmatcher_LIBRARIES}
)
