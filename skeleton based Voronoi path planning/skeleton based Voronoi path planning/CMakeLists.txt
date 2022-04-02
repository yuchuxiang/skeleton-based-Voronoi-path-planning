cmake_minimum_required(VERSION 3.21)
project(find_point)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

add_executable(find_point find_point.cpp)
target_link_libraries(find_point ${OpenCV_LIBS})
