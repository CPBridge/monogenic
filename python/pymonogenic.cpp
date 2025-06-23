#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>

#include "cvnp/cvnp.h"
#include "monogenic/monogenicProcessor.h"

namespace py = pybind11;

// Forward declaration for cvnp registration function if needed
void pydef_cvnp(pybind11::module& m); // Removed - not needed if using cvnp:: namespace

PYBIND11_MODULE(pymonogenic , m) {
    // Optional: Add a docstring for your module
    m.doc() = "Python bindings for the monogenic signal processing library";

    // Expose the monogenicProcessor class (keeping this allows users
    // to use the class-based interface if they prefer, or if they need
    // other methods like getFeatureSymmetry, getEvenFilt, etc.)
    py::class_<monogenic::monogenicProcessor>(m, "MonogenicProcessor")
        // Expose the constructor
        .def(py::init<const int, const int, const float, const float, const float>(),
             "Constructor for MonogenicProcessor.",
             py::arg("image_size_y"),
             py::arg("image_size_x"),
             py::arg("wavelength"),
             py::arg("shape_sigma") = 0.5,
             py::arg("sym_thresh") = 0.16
        )
        // Assuming findMonogenicSignal is needed for the class interface
        .def("findMonogenicSignal",
             [](monogenic::monogenicProcessor& self, const cv::Mat& input_image) {
                 self.findMonogenicSignal(input_image);
             },
             "Calculates the monogenic signal for the input image.",
             py::arg("input_image")
        )
        // Expose the getFeatureAsymmetry method as a class method
        .def("getFeatureAsymmetry",
            [](monogenic::monogenicProcessor& self) {
                cv::Mat asymmetry_image;
                self.getFeatureAsymmetry(asymmetry_image);
                return cvnp::mat_to_nparray(asymmetry_image);
            },
            "Gets the feature asymmetry image after findMonogenicSignal has been called. Returns a NumPy array."
        )

        // Expose the getFeatureAsymmetry method as a class method
        .def("getFeatureSymmetry",
            [](monogenic::monogenicProcessor& self) {
                cv::Mat symmetry_image;
                self.getFeatureSymmetry(symmetry_image);
                return cvnp::mat_to_nparray(symmetry_image);
            },
            "Gets the feature symmetry image after findMonogenicSignal has been called. Returns a NumPy array."
        )
        
        .def("getEvenFilt",
            [](monogenic::monogenicProcessor& self) {
                cv::Mat even_filter_image;
                self.getEvenFilt(even_filter_image);
                return cvnp::mat_to_nparray(even_filter_image);
            },
            "Gets the even part of the monogenic representation. Returns a NumPy array."
        )

        .def("getOddFiltCartesian",
            [](monogenic::monogenicProcessor& self) {
                cv::Mat odd_filter_y_image;
                cv::Mat odd_filter_x_image;
                self.getOddFiltCartesian(odd_filter_y_image, odd_filter_x_image);
                return  py::make_tuple(cvnp::mat_to_nparray(odd_filter_y_image), cvnp::mat_to_nparray(odd_filter_x_image));
            },
            "Gets the odd part of the monogenic representation. Returns 2 NumPy arrays."
        )

        .def("getSignedSymmetry",
            [](monogenic::monogenicProcessor& self) {
                cv::Mat pos_fs;
                cv::Mat neg_fs;
                self.getSignedSymmetry(pos_fs, neg_fs);
                return  py::make_tuple(cvnp::mat_to_nparray(pos_fs), cvnp::mat_to_nparray(neg_fs));
            },
            "Gets the oriented symmetry as two separate images (one for positive symmetry and the other for negative symmetry). Returns 2 NumPy arrays."
        )

        .def("getOrientedAsymmetry",
            [](monogenic::monogenicProcessor& self) {
                cv::Mat fa;
                cv::Mat lo;
                self.getOrientedAsymmetry(fa, lo);
                return  py::make_tuple(cvnp::mat_to_nparray(fa), cvnp::mat_to_nparray(lo));
            },
            "Gets the oriented asymmetry as two separate images (one for magnitude and the other for orientation). Returns 2 NumPy arrays."
        )

        ;

    // --- Expose a combined function for direct asymmetry calculation ---
    // This function takes the input image and constructor parameters,
    // handles object creation and method calls internally, and returns the result.
    m.def("compute_feature_asymmetry",
          [](const cv::Mat& input_image,
             const float wavelength,
             const float shape_sigma = 0.5,
             const float sym_thresh = 0.16) {

              // Get image dimensions from the input cv::Mat
              int image_size_y = input_image.rows;
              int image_size_x = input_image.cols;

              // 1. Create a MonogenicProcessor object
              monogenic::monogenicProcessor processor(image_size_y, image_size_x,
                                                      wavelength, shape_sigma, sym_thresh);

              // 2. Calculate the monogenic signal for the input image
              processor.findMonogenicSignal(input_image);

              // 3. Get the feature asymmetry image
              cv::Mat asymmetry_image;
              processor.getFeatureAsymmetry(asymmetry_image);

              // 4. Convert the resulting cv::Mat to a NumPy array using cvnp
              return cvnp::mat_to_nparray(asymmetry_image);
          },
          "Computes the feature asymmetry for an image using the monogenic signal. "
          "Combines object creation, signal calculation, and result retrieval.",
          py::arg("input_image"),
          py::arg("wavelength"), // Corrected arg order
          py::arg("shape_sigma") = 0.5,
          py::arg("sym_thresh") = 0.16
    );


    // Expose cvnp's conversion functions to the module
    pydef_cvnp(m);
}
