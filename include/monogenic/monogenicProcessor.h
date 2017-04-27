#ifndef MONOGENICFEATEXTRACTOR_H
#define MONOGENICFEATEXTRACTOR_H
#include <opencv2/core/core.hpp>

namespace monogenic
{

class monogenicProcessor
{
	public:

	// Simple constructor
	monogenicProcessor();

	// Full constructor
	// You must provide the image dimensions and wavelength
	// You may choose the specify shape parameter of the log-Gabor filter used
	// to calculate the monogenic signal and the threshold parameter to use for
	// feature symmetry and asymmetry calculations
	monogenicProcessor(const int image_size_y, const int image_size_x, const float wavelength, const float shape_sigma = 0.5, const float sym_thresh = 0.16); // constructor

	// Reinitialise an object, parameter as in constructor
	void initialise(const int image_size_y, const int image_size_x, const float wavelength, const float shape_sigma = 0.5, const float sym_thresh = 0.16);

	// Calculate the monogenic representation of the input image I
	// This does not return anything, it just stores the result for use in future
	// queries. It must be called before using any of the following methods to
	// obtain results
	// It also overwrites any previous result
	void findMonogenicSignal(const cv::Mat &I);

	// Returns the even part of the monogenic representation
	void getEvenFilt(cv::Mat &even);

	// Returns a polar representation (magnitude and local orientation) of the
	// odd part of the monogenic representation
	void getOddFiltPolar(cv::Mat &mag, cv::Mat &lo);

	// Returns a Cartesian representation (x and y components) of the
	// odd part of the monogenic representation
	void getOddFiltCartesian(cv::Mat &odd_y, cv::Mat &odd_x);

	// Returns a complex-valued representation of the
	// odd part of the monogenic representation
	void getOddFiltComplex(cv::Mat &odd);

	// Returns the feature symmetry measure calculated from the monogenic
	// representation
	void getFeatureSymmetry(cv::Mat &fs);

	// Returns the feature asymmetry measure calculated from the monogenic
	// representation
	void getFeatureAsymmetry(cv::Mat &fa);

	// Returns the signed feature symmetry measure calculated from the monogenic
	// representation. This consists of separate positive (peaks) and negative
	// parts
	void getSignedSymmetry(cv::Mat &pos_fs, cv::Mat &neg_fs);

	// Returns the oriented feature asymmetry measure calculated from the monogenic
	// representation. This consists of the magnitude of the feature symmetry (fa)
	// and the local orientation (lo)
	void getOrientedAsymmetry(cv::Mat &fa, cv::Mat &lo);

	// Returns the local phase measure calculated from the monogenic
	// representation.
	void getLocalPhase(cv::Mat &lp);

	// Returns the local phase vector measure calculated from the monogenic
	// representation. This consists of the magnitude of the local phase and the
	// local orientation
	void getLocalPhaseVector(cv::Mat &mag, cv::Mat &lo);

	private:
	// Methods
	void createLogGaborRieszFilt(void);
	void splitEven();
	void splitOdd();
	void findEvenMag();
	void findOddMagOri();
	void findAmp();
	void findSym();
	void findOrSym();
	void findAsym();
	void findLP();

	// Data
	cv::Mat even_im_cmplx, odd_im_cmplx, even_im, odd_ims[2], even_mag, odd_mag, amp, sym, asym, pos_sym, neg_sym, ori, lp;
	bool even_valid, odd_valid, even_mag_valid, odd_mag_ori_valid, amp_valid, sym_valid, asym_valid, or_sym_valid, or_asym_valid, lp_valid;
	cv::Mat even_filter, odd_filter;
	cv::Mat planes[2];
	int ysize, xsize, pad_ysize, pad_xsize;
	float sigma_onf, wl, T;

	static constexpr float C_EPSILON = 0.0001; // small constant to avoid dividing by zero

};

} // end of namespace

#endif
