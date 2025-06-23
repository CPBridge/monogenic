# Monogenic Signal (OpenCV Implementation)

This is a basic implementation of the monogenic signal for 2D images using
the C++ language and the OpenCV library. As well the monogenic signal, several
related quantities that can be derived from the monogenic signal, such as Feature
Symmetry and Asymmetry, are also implemented.

The monogenic signal is an alternative way of representing an image, which has a
number of advantages for further processing. For an introduction to the monogenic
signal and derived features with references to the relevant scientific literature,
please see [this document](https://chrisbridge.science/docs/intro_to_monogenic_signal.pdf) (PDF). If you find this or the monogenic library useful consider citation of the [work](https://arxiv.org/pdf/1703.09199).

**Python bindings are also provided** using [pybind11](https://github.com/pybind/pybind11) & [cvnp](https://github.com/pthom/cvnp), allowing you to use this library from Python with seamless NumPy integration.

### Capabilities

Functions are provided to calculate the following quantities for 2D images:

* Monogenic Signal.
* Local Energy, Local Phase and Local Orientation to describe the local properties of image.
* Feature Symmetry and Asymmetry, respond to symmetric 'blobs' and boundaries with robustness to variable contrast.
* Oriented Feature Symmetry and Asymmetry, as above but also containing the polarity of the symmetry and the orientation of the boundaries.

This implementation was written with computational efficiency as a key objective,
such that it can be used for video processing applications. It is designed to avoid
redundant calculations when several quantities are desired from the same input
image.

However, it is also straightforward and appropriate to use for calculating single
quantities from still images.

### Dependencies

#### C++ Dependencies
* A C++ compiler supporting the C++14 standard (requires a relatively modern version of your compiler).
* The [OpenCV](http://opencv.org) library. Tested on version 4.2 and 4.5.5 but most fairly recent
versions should be compatible. If you are using GNU/Linux, there will probably
be a suitable packaged version in your distribution's repository.
* (Optional) If you use a C++ compiler supporting the
[OpenMP](http://openmp.org/wp/) standard (includes most major compilers on major
platforms including MSVC, g++ and clang) there may be a small speed boost due to
parallelisation.

#### Python Dependencies (for Python bindings)
* Python 3.6 or higher
* NumPy
* OpenCV Python (`cv2` installed via `pip install opencv-python`)
* The project uses Pybind11 and cvnp as submodules (see building instructions below)

### Building the Project

#### Initialize Submodules

**Important**: Before building, you must initialize the git submodules:

```bash
git submodule update --init --recursive
```

This will download the required Pybind11 and cvnp dependencies needed for the Python bindings.

#### Build with CMake

To build the project, use CMake with the `BUILD_PYTHON_BINDINGS` option on/off to add python bindings, by default it is disabled:

```bash
mkdir build
cd build
cmake -DBUILD_PYTHON_BINDINGS=ON ..
make -j$(nproc)
```

This will build:
- The monogenic C++ library
- The C++ example executable (`monogenic_image_test`)
If python bindings are enabled:
- The Python module (`pymonogenic`)

### Instructions for Use

#### C++ Usage

The implementation consists of a single C++ class (`monogenicProcessor`), defined
in the `src/monogenicProcessor.cpp` and `include/monogenic/monogenicProcessor.h`
files. To use the code in your project, you just need to include the `.cpp`
file in the usual way, and add the repository's `include/` directory in the
include path.

There is an example program showing how to use the class in the `example/cpp/`
directory. The comments in this file should demonstrate the basic usage.

#### Python Usage

After building the project, the Python module `pymonogenic` will be available. Check that using `make install` it is successfully the library is in interpreter's site-packages folder or use the enviromental variable `PYTHONPATH` to add the path to the build folder that contains the compiled pymonogenic library.

 You can use it like this:

```python
import cv2
import pymonogenic

# Load an image
image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

# Create a processor
processor = pymonogenic.MonogenicProcessor(image.shape[0], image.shape[1], wavelength=50.0)

# Calculate monogenic signal
processor.findMonogenicSignal(image)

# Get results
feature_asymmetry = processor.getFeatureAsymmetry()
feature_symmetry = processor.getFeatureSymmetry()
even_part = processor.getEvenFilt()
odd_y, odd_x = processor.getOddFiltCartesian()
```

### Running the Examples

#### C++ Example

To compile and run the C++ example:

```bash
# Build the project first (see Building section above)
cd build

# Run the example with a video file
./monogenic_image_test path/to/your/video_file.avi
```

#### Python Example

To run the Python example:

```bash
# Make sure you've built the project first
cd example/python

# Run with an image file
python monogenicImageTest.py path/to/your/image.jpg
```

The Python example will:
- Load and process the specified image
- Calculate all monogenic signal components
- Display the results in separate windows (Even filter, Odd Y, Odd X, Feature Symmetry, Feature Asymmetry)
- Automatically scale the display windows to fit your screen

### Author

Written by [Christopher Bridge](https://chrisbridge.science/) at the
University of Oxford's Institute of Biomedical Engineering.
