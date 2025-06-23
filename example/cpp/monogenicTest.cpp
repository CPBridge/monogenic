#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "monogenic/monogenicProcessor.h"
#include <iostream>

// This test file demonstrates the basic usage of the monogenicProcessor
// class. It expects as the only command line argument, the name of a single
// .avi video file. It calculates the 2D monogenic signal representation of each
// frame in the file and displays the even part and the two odd parts on the
// screen. It also calculates the feature symmetry and asymmetry measures,
// and displays these too

// The monogenic signal is a representation of single channel (greyscale) images,
// so if the supplied video is colour, it will be converted to greyscale before
// processing.

// Namespaces
using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
	// Check that the user has supplied a single argument, which is the video
	// file to use for the demonstration
	if( argc != 2)
	{
		cout <<" Usage: " << argv[0] << " <videofilename>" << endl;
		return -1;
	}
	const string vidname = argv[1];

	// Open the video file
	VideoCapture vid_obj;
	vid_obj.open(vidname);
	if ( !vid_obj.isOpened())
	{
		cout  << "Could not open reference " << vidname << endl;
		return -1;
	}

	// Get proportions of the video frames and their number
	const int n_frames = vid_obj.get(cv::CAP_PROP_FRAME_COUNT);
	const int xsize = vid_obj.get(cv::CAP_PROP_FRAME_WIDTH);
	const int ysize = vid_obj.get(cv::CAP_PROP_FRAME_HEIGHT);

	// Create three display windows, one for the even part, and one for the each
	// of the off parts
	namedWindow( "Even", WINDOW_AUTOSIZE );// Create a window for display.
	namedWindow( "Odd Y", WINDOW_AUTOSIZE );// Create a window for display.
	namedWindow( "Odd X", WINDOW_AUTOSIZE );// Create a window for display.
	namedWindow( "Feature Symmetry", WINDOW_AUTOSIZE );// Create a window for display.
	namedWindow( "Feature Asymmetry", WINDOW_AUTOSIZE );// Create a window for display.

	// The first step to using the monogenicProcessor is to initialise
	// a monogenic processor object. At a minimum we must provide the dimensions
	// of the input image and a centre-wavelength to use for the log Gabor filter.
	// The shorter the wavelength, the more fine detail is preserved.
	// We'll choose 50 pixels, for no particular reason
	// Once created this object must only be used with images of the matching
	// size
	monogenic::monogenicProcessor mgFilts(ysize, xsize, 50);

	// Loop to cycle through frames
	Mat I, even, oddy, oddx, fs, fa, disp1, disp2, disp3, disp4, disp5;
	for(int f = 0 ; f < n_frames; ++f)
	{
		// Grab next frame and convert to greyscale
		vid_obj >> I;
		cvtColor(I,I,cv::COLOR_BGR2GRAY);

		// This line performs the calculation on the image I to find the monogenic
		// signal representation and must be performed before trying to access
		// the components of the monogenic signal or any of the derived measures
		// (such as feature symmetry)
		mgFilts.findMonogenicSignal(I);

		// Now we can access the odd and even components of the monogenic signal
		// representation of I
		mgFilts.getEvenFilt(even);
		mgFilts.getOddFiltCartesian(oddy,oddx);

		// We can also access some quantities derived from the monogenic signal,
		// such as feature symmetry (fs) and feature asymmetry (fa).
		mgFilts.getFeatureSymmetry(fs);
		mgFilts.getFeatureAsymmetry(fa);

		// Display even and odd components
		normalize(even,disp1,0,1, cv::NORM_MINMAX);
		imshow("Even", disp1);
		normalize(oddy,disp2,0,1, cv::NORM_MINMAX);
		imshow("Odd Y", disp2);
		normalize(oddx,disp3,0,1, cv::NORM_MINMAX);
		imshow("Odd X", disp3);

		// Display feature symmetry and asymmetry
		normalize(fs,disp4,0,1, cv::NORM_MINMAX);
		imshow("Feature Symmetry", disp4);
		normalize(fa,disp5,0,1, cv::NORM_MINMAX);
		imshow("Feature Asymmetry", disp5);

		// Pause the loop for short while
		waitKey(10);
	}
}
