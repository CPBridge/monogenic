cmake_minimum_required(VERSION 3.15)
project(pymonogenic)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(pymonogenic pymonogenic.cpp)

set_target_properties(pymonogenic PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

target_include_directories(pymonogenic PRIVATE
    ${CMAKE_SOURCE_DIR}/include          # Main monogenic headers
    ${CMAKE_SOURCE_DIR}/src              # Source directory
    ${CMAKE_SOURCE_DIR}/cvnp/cvnp        # cvnp headers
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(pymonogenic PRIVATE
    pybind11::module                     # Use module, not embed
    cvnp
    monogenic
    ${OpenCV_LIBS}
)

# Compiler-specific options
target_compile_definitions(pymonogenic PRIVATE VERSION_INFO="${EXAMPLE_VERSION_INFO}")

# Install the Python module to the site-packages directory of the Python environment
# being used to build the library. This ensures the module is available for import
# in the same Python environment that was used during compilation.
#
# Alternatively, you can set the PYTHONPATH environment variable to the build directory 
# to make the module available for import in the same Python environment that was used during compilation.
install(TARGETS pymonogenic
    DESTINATION ${Python3_SITELIB}        
)

