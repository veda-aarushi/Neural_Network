cmake_minimum_required(VERSION 3.27)
project(Neural_Network)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(Neural_Network
        Neural_Network.cpp
        Neural_Network.h)
