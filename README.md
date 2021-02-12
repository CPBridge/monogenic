# Monogenic Signal (OpenCV Implementation)

This is a basic implementation of the monogenic signal for 2D images using
the C++ language and the OpenCV library. As well the monogenic signal, several
related quantities that can be derived from the monogenic signal, such as Feature
Symmetry and Asymmetry, are also implemented.

The monogenic signal is an alternative way of representing an image, which has a
number of advantages for further processing. For an introduction to the monogenic
signal and derived features with references to the relevant scientific literature,
please see [this document](https://chrisbridge.science/docs/intro_to_monogenic_signal.pdf) (PDF).

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

* A C++ compiler supporting the C++11 standard (requires a relatively modern version of your compiler).
* The [OpenCV](http://opencv.org) library. Tested on version 4.2 but most fairly recent
versions should be compatible. If you are using GNU/Linux, there will probably
be a suitable packaged version in your distribution's repository.
* (Optional) If you use a C++ compiler supporting the
[OpenMP](http://openmp.org/wp/) standard (includes most major compilers on major
platforms including MSVC, g++ and clang) there may be a small speed boost due to
parallelisation.

### Instructions for Use

The implementation consists of a single C++ class (`monogenicProcessor`), defined
in the `src/monogenicProcessor.cpp` and `include/monogenic/monogenicProcessor.h`
files. To use the code in your project, you just need to include the `.cpp`
file in the usual way, and add the repository's `include/` directory in the
include path.

There is an example programme showing how to use the class in the `example/`
directory. The comments in this file should demonstrate the basic usage.

### Compiling and Running the Example

To compile the example on a GNU/Linux system, simply run the `make` command from
the `example/` directory. To run the example, then execute

```bash
$ ./monogenicTest video_file.avi
```

where `video_file.avi` is the name of a video file. This will then calculate
the monogenic signal and feature symmetry and asymmetry images, and display then.

### Author

Written by [Christopher Bridge](https://chrisbridge.science/) at the
University of Oxford's Institute of Biomedical Engineering.
