# This is a stub CMakeLists.txt file for the monogenic project
# It provides basic configuration for building the monogenic library with OpenCV dependencies

cmake_minimum_required(VERSION 3.22)
project(monogenic CXX)

# Set the C++ standard for your wrapper project
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Build options
option(BUILD_PYTHON_BINDINGS "Build Python bindings for monogenic" OFF)

# Find required packages first
find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs imgproc)

# Find Python only if building Python bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
endif()

# Include directories after finding packages
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include # Include monogenic library headers
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Create the monogenic library
add_library(monogenic src/monogenicProcessor.cpp)
target_link_libraries(monogenic ${OpenCV_LIBS})

# Make sure the monogenic library includes the necessary headers
target_include_directories(monogenic PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Create the test executable
add_executable(monogenic_image_test ${CMAKE_CURRENT_SOURCE_DIR}/example/cpp/monogenicTest.cpp)

target_include_directories(monogenic_image_test PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(monogenic_image_test PUBLIC
    monogenic         # Link to the monogenic library target
    ${OpenCV_LIBS}    # Link to the necessary OpenCV libraries
)

# Add Python-related subdirectories only if building Python bindings
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(cvnp)
    add_subdirectory(pybind11)
    add_subdirectory(python)
endif()