cmake_minimum_required(VERSION 2.8.3)
project(feature_based_detector)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  message_filters
)

find_package(OpenCV REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES feature_based_detector
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_definitions(-std=c++11)

add_executable(feature_based_detector_node src/feature_based_detector_node.cpp)

target_link_libraries(feature_based_detector_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)