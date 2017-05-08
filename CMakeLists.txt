# You may use this CMake file to include the RIFeatures library in other
# projects using CMake

find_package( OpenCV REQUIRED )
add_library(monogenic src/monogenicProcessor.cpp)
target_link_libraries( monogenic ${OpenCV_LIBS} )
