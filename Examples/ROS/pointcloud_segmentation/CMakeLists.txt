cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

rosbuild_init()

IF(NOT ROS_BUILD_TYPE)
  SET(ROS_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${ROS_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../cmake_modules)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

set(Eigen3_DIR /usr/local/include/eigen3)
find_package(Boost REQUIRED python3)
find_package(Eigen3 3.1.0 REQUIRED)

# adding for point cloud viewer and mapper
find_package( PCL 1.7 REQUIRED )
find_package(catkin REQUIRED COMPONENTS 
  roscpp
  rospy
  sensor_msgs
  message_generation
  tf
  )

# include_directories(
# ${PROJECT_SOURCE_DIR}
# ${PCL_INCLUDE_DIRS}
# ${catkin_INCLUDE_DIRS}
# )

set(CATKIN_PACKAGE_BIN_DESTINATION ${PROJECT_SOURCE_DIR}/bin)
catkin_install_python(PROGRAMS src/image_segmentation.py
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# set(LIBS 
# ${OpenCV_LIBS} 
# ${EIGEN3_LIBS}
# ${PCL_LIBRARIES}
# ${catkin_LIBRARIES}
# -lboost_system
# )
