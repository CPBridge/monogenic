#include "monogenic/monogenicProcessor.h"
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

namespace monogenic
{

// Simple constructor without initialisation
monogenicProcessor::monogenicProcessor()
{
}

// Constructor with initialisation
monogenicProcessor::monogenicProcessor(const int image_size_y, const int image_size_x, const float wavelength, const float shape_sigma, const float sym_thresh)
{
	initialise(image_size_y,image_size_x,wavelength,shape_sigma,sym_thresh);
}

// Constructor
void monogenicProcessor::initialise(const int image_size_y, const int image_size_x, const float wavelength, const float shape_sigma, const float sym_thresh)
{
	// Copy input parameters
	xsize = image_size_x;
	ysize = image_size_y;
	wl = wavelength;
	sigma_onf = shape_sigma;
	T = sym_thresh;

	// Set up parameters for Fourier transforming the incoming images
	pad_xsize = getOptimalDFTSize(xsize);
	pad_ysize = getOptimalDFTSize(ysize);
	planes[1] = Mat::zeros(pad_ysize,pad_xsize,CV_32F);

	// Create the monogenic filters
	createLogGaborRieszFilt();

	// Set all flags to false
	even_valid = false;
	odd_valid = false;
	even_mag_valid = false;
	odd_mag_ori_valid = false;
	amp_valid = false;
	sym_valid = false;
	asym_valid = false;
	or_sym_valid = false;
	or_asym_valid = false;
	lp_valid = false;
}

// Function to construct a log Gabor filter (even) and its
// complex-valued Riesz transform
void monogenicProcessor::createLogGaborRieszFilt(void)
{

	MatIterator_<Vec2f> even_it, odd_it, even_end;
	int i = 0, j = 0;
	float w_x, w_y, w, f;
	const float w0 = 1.0/wl;
	const float scale_const = 1.0/(2.0*std::log(sigma_onf)*std::log(sigma_onf));
	const float xsizef = float (pad_xsize);
	const float ysizef = float(pad_ysize);

	// Find coordinates at which we switch to negative frequencies
	const int xswitch = (pad_xsize % 2 == 0) ? pad_xsize/2 : (pad_xsize+1)/2;
	const int yswitch = (pad_ysize % 2 == 0) ? pad_ysize/2 : (pad_ysize+1)/2;

	// Set filters to zero
	even_filter= Mat::zeros(pad_ysize,pad_xsize, CV_32FC2);
	odd_filter = Mat::zeros(pad_ysize,pad_xsize, CV_32FC2);

	// Iterate through pixels of the filter
	even_end = even_filter.end<Vec2f>();
	for (even_it = even_filter.begin<Vec2f>(), odd_it = odd_filter.begin<Vec2f>(); even_it != even_end; ++even_it, ++odd_it)
	{
		// Find freq value of this coordinate
		w_x = (i < xswitch) ? float(i)/xsizef : (float(i)-xsizef)/xsizef;
		w_y = (j < yswitch) ? float(-j)/ysizef : (ysizef - float(j))/ysizef;
		w = std::sqrt(w_x*w_x + w_y*w_y);

		// Calculate filter values
		if (!((i == 0) && (j == 0)))
		{
			// In an even-dimension, we need to zero the highest frequency component (as this is unpaired)
			if(((pad_xsize % 2 == 0) && (i == xswitch))
				|| ((pad_ysize % 2 == 0) && (j == yswitch)) )
				f = 0.0;
			else
				f = std::exp( (-std::pow(std::log(w/w0),2.0) ) * scale_const); // lg filter
			(*even_it)[0] = f;
			(*odd_it)[0] = -f*w_y/w; // complex reisz filter
			(*odd_it)[1] = f*w_x/w;
		}

		// Find coordinates of the next pixel
		++i;
		if (i == pad_xsize)
		{
			i = 0;
			 ++j;
		}
	}
}

// This function is used to input a new image. The even and odd filter responses are found
// via the DFT, and other images are invalidated.
void monogenicProcessor::findMonogenicSignal(const Mat &I)
{
	Mat temp, padded;

	// Make sure the input image is greyscale
	if(I.channels() == 3)
	{
		cvtColor(I,padded,cv::COLOR_BGR2GRAY);
		copyMakeBorder(padded, padded, 0, pad_ysize - ysize, 0, pad_xsize - xsize, BORDER_CONSTANT, Scalar::all(0));  //expand input image to optimal size
	}
	else
	{
		// Pad the input image
		copyMakeBorder(I, padded, 0, pad_ysize - ysize, 0, pad_xsize - xsize, BORDER_CONSTANT, Scalar::all(0));  //expand input image to optimal size
	}

	padded.convertTo(planes[0],CV_32F);
	merge(planes, 2, temp);

	// Take the DFT
	dft(temp,temp);

	// Perform odd and even calculations in parallel
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			// Use the even filter
			mulSpectrums(temp,even_filter,even_im_cmplx,0);
			idft(even_im_cmplx,even_im_cmplx,DFT_SCALE);
		}
		#pragma omp section
		{
			// Use the odd filter
			mulSpectrums(temp,odd_filter,odd_im_cmplx,0);
			idft(odd_im_cmplx,odd_im_cmplx,DFT_SCALE);
		}
	}

	// Set all flags to false
	even_valid = false;
	odd_valid = false;
	even_mag_valid = false;
	odd_mag_ori_valid = false;
	amp_valid = false;
	sym_valid = false;
	asym_valid = false;
	or_sym_valid = false;
	or_asym_valid = false;
	lp_valid = false;
}

// Calculates and stores feature symmetry, and any dependencies if
// necessary
void monogenicProcessor::findSym()
{
	Mat temp;

	if(!amp_valid) findAmp();

	temp = even_mag - odd_mag - T ;
	threshold(temp,temp,0,0,THRESH_TOZERO);
	sym = temp / (amp + C_EPSILON);

	sym_valid = true;
}

// Calculates and stores feature asymmetry, and any dependencies if
// necessary
void monogenicProcessor::findAsym()
{
	Mat temp;

	if(!amp_valid) findAmp();

	temp = odd_mag - even_mag - T ;
	threshold(temp,temp,0,0,THRESH_TOZERO);
	asym = temp / (amp + C_EPSILON);

	asym_valid = true;
}

// Calculates and stores oriented feature symmetry, and any dependencies
// if necessary
void monogenicProcessor::findOrSym()
{
	Mat temp;

	if(!amp_valid) findAmp();

	// Positive symmetry
	threshold(even_im,temp,0.0,0,THRESH_TOZERO);
	temp = temp - odd_mag - T ;
	threshold(temp,temp,0,0,THRESH_TOZERO);
	pos_sym = temp / (amp + C_EPSILON);

	// Negative symmetry
	threshold(-even_im,temp,0.0,0,THRESH_TOZERO);
	temp = temp - odd_mag - T ;
	threshold(temp,temp,0,0,THRESH_TOZERO);
	neg_sym = temp / (amp + C_EPSILON);

	or_sym_valid = true;
}

// Split the even response into two planes (the imaginary plane can be
// as it should be zero if not for numerical errors)
void monogenicProcessor::splitEven()
{
	Mat planes[2];
	split(even_im_cmplx,planes);
	even_im = planes[0];
	even_valid = true;
}

// Split the complex odd response into two planes (the two directions)
// and store
void monogenicProcessor::splitOdd()
{
	split(odd_im_cmplx,odd_ims);
	odd_valid = true;
}

// Find and store the magnitude (absolute value) of the even filter
// response
void monogenicProcessor::findEvenMag()
{
	if(!even_valid) splitEven();
	even_mag = cv::abs(even_im);
	even_mag_valid = true;
}

// Find and store the magnitude and orientation of the odd filter
void monogenicProcessor::findOddMagOri()
{
	if(!odd_valid) splitOdd();
	cartToPolar(odd_ims[0],odd_ims[1],odd_mag,ori);
	odd_mag_ori_valid = true;
}

// Find and store the local amplitude
void monogenicProcessor::findAmp()
{
	if(!even_mag_valid) findEvenMag();
	if(!odd_mag_ori_valid) findOddMagOri();
	magnitude(odd_mag,even_mag,amp);
	amp_valid = true;
}

// Find and store the local phase
void monogenicProcessor::findLP()
{
	if(!odd_mag_ori_valid) findOddMagOri();
	if(!even_valid) splitEven();
	phase(even_im,odd_mag,lp);
	lp_valid = true;
}

// (Calculates and) Returns the even filter response
void monogenicProcessor::getEvenFilt(Mat &even)
{
	if(!even_valid) splitEven();
	even = even_im;
}

// (Calculates and) Returns the odd response as two separate images
// (one for magnitude and the other for orientation)
void monogenicProcessor::getOddFiltPolar(Mat &mag, Mat &lo)
{
	if(!odd_mag_ori_valid) findOddMagOri();
	mag = odd_mag;
	lo = ori;
}

// (Calculates and) Returns the odd response as two separate images
// (one for each axis direction)
void monogenicProcessor::getOddFiltCartesian(Mat &odd_y, Mat &odd_x)
{
	if(!odd_valid) splitOdd();
	odd_y = odd_ims[1];
	odd_x = odd_ims[0];
}

// Returns the odd response as a single complex (two-channeled) image
void monogenicProcessor::getOddFiltComplex(Mat &odd)
{
	odd = odd_im_cmplx;
}

// (Calculates and) Returns the feature symmetry
void monogenicProcessor::getFeatureSymmetry(Mat &fs)
{
	if(!sym_valid) findSym();
	fs = sym;
}

// (Calculates and) Returns the feature asymmetry
void monogenicProcessor::getFeatureAsymmetry(Mat &fa)
{
	if(!asym_valid) findAsym();
	fa = asym;
}

// (Calculates and) Returns the oriented symmetry as two separate images
// (one for positive symmetry and the other for negative symmetry)
void monogenicProcessor::getSignedSymmetry(Mat &pos_fs, Mat &neg_fs)
{
	if(!or_sym_valid) findOrSym();
	pos_fs = pos_sym;
	neg_fs = neg_sym;
}

// (Calculates and) Returns the oriented asymmetry as two separate images
// (one for magnitude and the other for orientation)
void monogenicProcessor::getOrientedAsymmetry(Mat &fa, Mat &lo)
{
	if(!asym_valid) findAsym();
	fa = asym;
	lo = ori;
}


void monogenicProcessor::getLocalPhase(Mat &lp)
{
	if(!lp_valid) findLP();
	lp = this->lp;
}

void monogenicProcessor::getLocalPhaseVector(Mat &mag, Mat &lo)
{
	if(!lp_valid) findLP();
	mag = lp;
	lo = ori;
}

} // end of namespace
