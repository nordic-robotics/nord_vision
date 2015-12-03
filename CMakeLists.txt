cmake_minimum_required(VERSION 2.8.3)
project(nord_vision)
set(CMAKE_CXX_FLAGS "--std=c++11 ${CMAKE_CXX_FLAGS}")

find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_geometry 
  std_msgs 
  roscpp 
  rospy
  image_transport
  sensor_msgs
  nord_messages  
  ras_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  nord_messages
  ras_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

#add_executable(closestpoint src/test.cpp)
#target_link_libraries(closestpoint ${catkin_LIBRARIES} ${OpenCV_LIBS})
