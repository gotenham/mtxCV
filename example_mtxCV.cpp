#include "mtxCV.h"

#define IMAGE_WIDTH 32
#define IMAGE_HEIGHT 24

#define IMG_ARR_SIZE 768

using namespace std; //< Standard lib: vector, cout etc...
using namespace Eigen; //< Eigen vector and matrix manipulation

const uint8_t gaussian_N = 5; // SET Gaussian NxN kernel size, must be odd

const float cannyGradientThreshHigh = 0.7f; // ratio of largest gradient in suppressed image; edge must have gradient >= this to qualify as strong edge
const float cannyGradientThreshLow = 0.3f; // ratio of {cannyGradientThreshHigh OR largest gradient in suppressed image}, edge must have gradient >= this to qualify a weak edge
const size_t minSequenceLength = 10; //minimum sequence of edge points required to qualify being added to edgeSequenceListReturn
const uint8_t strongEdgeValue = 255; // candidate edges are assigned this value in the canny returned edgeMask when >= calculated dynThreshHigh
const uint8_t weakEdgeValue = 160; // candidate edges are assigned this value in the canny returned edgeMask when >= calculated dynThreshLow
vector<EdgePoint> edgeCandidates; // init var to store coordinates of identified weak and strong edges in cannyEdgeDetection returned image edge mask

static Contours allContours; // init storage for list of identified edge sequences


// example image frame init
static MatrixXi mtxImageFrame(IMAGE_HEIGHT, IMAGE_WIDTH); //example to store processed image
static MatrixXi edgeMask(IMAGE_HEIGHT, IMAGE_WIDTH); //example to store image edge mask

int main() {
	FnState fnState; // tracks state of mtxCV function result, feel free to use as applicable

	// OPTION 1: example kernel init using full 2D matrix
	//static Matrix<float, (int)gaussian_N, (int)gaussian_N> gaussianKernel; //example to store gaussian NxN kernel
	// Example generating gaussian NxN kernel into gaussianKernel
	//fnState = MtxCV::generateGaussianKernel<gaussian_N>(gaussianKernel);

	// OPTION 2: example init seperable kernel row and col vect for seperable convolution
	static RowVector<float, (int)gaussian_N> gaussianRowVector; //example to store gaussian NxN kernel
	static Vector<float, (int)gaussian_N> gaussianColVector; //example to store gaussian NxN kernel
	// Example generating seperable gaussian kernel into gaussianColVector and gaussianRowVector
	fnState = MtxCV::generateGaussianKernelSeperated<gaussian_N>(gaussianColVector, gaussianRowVector); 
	//NOTE: recommend calculating vectors external to microcontroller and hard coding result, as decomposition inclusion of Eigen::SVD libraries requires large memory


	//load 1D array as applicable, example data of earth partial image below
	float imgArr[IMG_ARR_SIZE] = {0.497f, 0.f, 0.1f, 0.1f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
	0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 6.497f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.624f, 0.f, 0.f, 6.497f, 0.f, 0.f, 0.f, 1.624f, 0.f, 0.f, 0.f, 0.f,
	1.624f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.624f, 0.f, 0.f, 0.f, 1.624f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 6.497f, 0.f,
	0.f, 1.624f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 6.497f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.624f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
	0.f, 6.497f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
	0.f, 0.f, 0.f, 0.f, 0.f, 6.497f, 0.f, 0.f, 1.624f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.624f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 6.497f, 0.f, 0.f, 0.f, 0.f, 0.f,
	0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
	0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
	0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.624f, 1.624f, 42.229f, 42.229f, 102.325f, 102.325f, 139.682f, 139.682f, 159.172f, 159.172f, 155.924f, 155.924f, 134.809f, 134.809f, 108.822f, 108.822f, 79.586f, 79.586f, 48.726f, 48.726f,
	0.f, 6.497f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.624f, 1.624f, 42.229f, 42.229f, 102.325f, 102.325f, 139.682f, 139.682f, 159.172f, 159.172f, 155.924f, 155.924f, 134.809f, 134.809f, 108.822f, 108.822f, 79.586f, 79.586f, 48.726f, 48.726f,
	0.f, 0.f, 1.624f, 0.f, 0.f, 0.f, 4.873f, 4.873f, 84.459f, 84.459f, 190.032f, 190.032f, 255.f, 255.f, 245.255f, 245.255f, 222.516f, 222.516f, 191.656f, 191.656f, 159.172f, 159.172f, 134.809f, 134.809f, 113.694f, 113.694f, 92.58f, 92.58f, 73.089f, 73.089f, 51.975f, 51.975f,
	0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 4.873f, 4.873f, 84.459f, 84.459f, 190.032f, 190.032f, 255.f, 255.f, 245.255f, 245.255f, 222.516f, 222.516f, 191.656f, 191.656f, 159.172f, 159.172f, 134.809f, 134.809f, 113.694f, 113.694f, 92.58f, 92.58f, 73.089f, 73.089f, 51.975f, 51.975f,
	0.f, 0.f, 16.242f, 16.242f, 100.701f, 100.701f, 198.153f, 198.153f, 235.51f, 235.51f, 216.019f, 216.019f, 162.42f, 162.42f, 147.803f, 147.803f, 154.299f, 154.299f, 128.312f, 128.312f, 100.701f, 100.701f, 92.58f, 92.58f, 87.707f, 87.707f, 66.592f, 66.592f, 45.478f, 45.478f, 42.229f, 42.229f,
	0.f, 0.f, 16.242f, 16.242f, 100.701f, 100.701f, 198.153f, 198.153f, 235.51f, 235.51f, 216.019f, 216.019f, 162.42f, 162.42f, 147.803f, 147.803f, 154.299f, 154.299f, 128.312f, 128.312f, 100.701f, 100.701f, 92.58f, 92.58f, 87.707f, 87.707f, 66.592f, 66.592f, 45.478f, 45.478f, 42.229f, 42.229f,
	50.35f, 50.35f, 115.318f, 115.318f, 141.306f, 141.306f, 175.414f, 175.414f, 157.548f, 157.548f, 128.312f, 128.312f, 103.949f, 103.949f, 99.076f, 99.076f, 81.21f, 81.21f, 69.841f, 69.841f, 63.344f, 63.344f, 47.102f, 47.102f, 37.357f, 37.357f, 32.484f, 32.484f, 22.739f, 22.739f, 19.49f, 19.49f,
	50.35f, 50.35f, 115.318f, 115.318f, 141.306f, 141.306f, 175.414f, 175.414f, 157.548f, 157.548f, 128.312f, 128.312f, 103.949f, 103.949f, 99.076f, 99.076f, 81.21f, 81.21f, 69.841f, 69.841f, 63.344f, 63.344f, 47.102f, 47.102f, 37.357f, 37.357f, 32.484f, 32.484f, 22.739f, 22.739f, 19.49f, 19.49f,
	50.35f, 50.35f, 82.834f, 82.834f, 134.809f, 134.809f, 105.573f, 105.573f, 79.586f, 79.586f, 79.586f, 79.586f, 66.592f, 66.592f, 45.478f, 45.478f, 73.089f, 73.089f, 71.465f, 71.465f, 38.981f, 38.981f, 29.236f, 29.236f, 21.115f, 21.115f, 16.242f, 16.242f, 9.745f, 9.745f, 4.873f, 4.873f,
	50.35f, 50.35f, 82.834f, 82.834f, 134.809f, 134.809f, 105.573f, 105.573f, 79.586f, 79.586f, 79.586f, 79.586f, 66.592f, 66.592f, 45.478f, 45.478f, 73.089f, 73.089f, 71.465f, 71.465f, 38.981f, 38.981f, 29.236f, 29.236f, 21.115f, 21.115f, 16.242f, 16.242f, 9.745f, 9.745f, 4.873f, 4.873f,
	32.484f, 32.484f, 55.223f, 55.223f, 60.096f, 60.096f, 38.981f, 38.981f, 42.229f, 42.229f, 56.847f, 56.847f, 45.478f, 45.478f, 29.236f, 29.236f, 40.605f, 40.605f, 29.236f, 29.236f, 21.115f, 21.115f, 17.866f, 17.866f, 14.618f, 14.618f, 8.121f, 8.121f, 3.248f, 3.248f, 6.497f, 6.497f,
	32.484f, 32.484f, 55.223f, 55.223f, 60.096f, 60.096f, 38.981f, 38.981f, 42.229f, 42.229f, 56.847f, 56.847f, 45.478f, 45.478f, 29.236f, 29.236f, 40.605f, 40.605f, 29.236f, 29.236f, 21.115f, 21.115f, 17.866f, 17.866f, 14.618f, 14.618f, 8.121f, 8.121f, 3.248f, 3.248f, 6.497f, 6.497f,
	14.618f, 14.618f, 11.369f, 11.369f, 12.994f, 12.994f, 19.49f, 19.49f, 32.484f, 32.484f, 30.86f, 30.86f, 24.363f, 24.363f, 21.115f, 21.115f, 24.363f, 24.363f, 17.866f, 17.866f, 8.121f, 8.121f, 8.121f, 8.121f, 4.873f, 4.873f, 1.624f, 1.624f, 1.624f, 1.624f, 3.248f, 3.248f,
	14.618f, 14.618f, 11.369f, 11.369f, 12.994f, 12.994f, 19.49f, 19.49f, 32.484f, 32.484f, 30.86f, 30.86f, 24.363f, 24.363f, 21.115f, 21.115f, 24.363f, 24.363f, 17.866f, 17.866f, 8.121f, 8.121f, 8.121f, 8.121f, 4.873f, 4.873f, 1.624f, 1.624f, 1.624f, 1.624f, 3.248f, 3.248f,
	6.497f, 6.497f, 4.873f, 4.873f, 4.873f, 4.873f, 6.497f, 6.497f, 8.121f, 8.121f, 11.369f, 11.369f, 9.745f, 9.745f, 8.121f, 8.121f, 6.497f, 6.497f, 6.497f, 6.497f, 3.248f, 3.248f, 3.248f, 3.248f, 4.873f, 4.873f, 3.248f, 3.248f, 1.624f, 1.624f, 0.f, 0.f,
	6.497f, 6.497f, 4.873f, 4.873f, 4.873f, 4.873f, 6.497f, 6.497f, 8.121f, 8.121f, 11.369f, 11.369f, 9.745f, 9.745f, 8.121f, 8.121f, 6.497f, 6.497f, 6.497f, 6.497f, 3.248f, 3.248f, 3.248f, 3.248f, 4.873f, 4.873f, 3.248f, 3.248f, 1.624f, 1.624f, 0.f, 0.f};

	
	MatrixXf mtxImageFrameNorm(IMAGE_HEIGHT, IMAGE_WIDTH); // Example init for normalised image return if applicable
	
	// Example loading 1D float array into matrix image frame, scaled from original range to uint8_t; 0 to 255
	fnState = MtxCV::getByteMatrixFromFloatArray<MatrixXi>(imgArr, IMG_ARR_SIZE, mtxImageFrame, mtxImageFrameNorm);
	
	// Example suppressing image intensities below 10% of intensity range
	fnState = MtxCV::suppressIntensityBelowRatio(mtxImageFrameNorm, 0.10f);

	// Example printing initial image frame to serial in ASCII, w/ aspect compensation, using asciiGradientShort range.
	printf("[Initial mtxImage Frame]:\n\r");
	MtxCV::printImageAsciiSerial(mtxImageFrame, true, false);
	
	// Example Re-OPTION 1: performing 2D convolution, eg gaussian blur with extended borders
	//fnState = MtxCV::convolution(mtxImageFrame, gaussianKernel, true);
	
	// Example Re-OPTION 2: performing 1D optimised convolution with seperable row and col vectors, eg gaussian blur with extended borders
	fnState = MtxCV::separableConvolution(mtxImageFrame, gaussianRowVector, gaussianColVector, true);
	
	// Example performing histogram equalisation
	fnState = MtxCV::equalizeHistogram(mtxImageFrame);
	
	// Example performing canny edge detection with non-max suppression
	fnState = MtxCV::cannyEdgeDetection(mtxImageFrame, edgeCandidates, cannyGradientThreshLow, cannyGradientThreshHigh, &edgeMask, weakEdgeValue, strongEdgeValue);
	
	// Example printing processed image to serial in ASCII, w/ aspect compensation, using asciiGradientShort range.
	printf("[Processed mtxImage Frame]:\n\r");
	MtxCV::printImageAsciiSerial(mtxImageFrame, true, false);
	// Example printing image edge mask of edge candidates identified by Canny in ASCII format, w/ aspect compensation, using asciiGradientShort range.
	printf("[Canny Edge Candidate Mask]:\n\r");
	MtxCV::printImageAsciiSerial(edgeMask, true, false);
	
	// Example performing edge tracking by hysteresis using Breadth First Search
	fnState = MtxCV::edgeTrackingByHysteresis(edgeMask, edgeCandidates, allContours, minSequenceLength, weakEdgeValue, strongEdgeValue);
	
	// Example getting the number of Edge sequences identified by edge tracking algorithm that are stored in allContours
	size_t numEdgeSequences = allContours.linkedEdges.size();
	printf("Found numEdgeSequences:[%d]\n\r", static_cast<int>(numEdgeSequences));
	
	// For each identified edge sequence in allContours
	printf("[Edge Sequence Masks]:\n\r");
	for (size_t i = 0; i < numEdgeSequences; ++i) {
		// Get the current vector<EdgePoint*> sequence in linkedEdges
		const std::vector<EdgePoint*>& edgeSequence = allContours.linkedEdges[i]; // auto& resolves as const vector<EdgePoint*>&
		// Example constructing image from each contour in list
		fnState = MtxCV::constructMatrixFromPoints(edgeSequence, edgeMask, true, &mtxImageFrame);
		
		// Example printing edge sequence to serial in ASCII, w/ aspect compensation, print as point sequence (all intensities > 0 are 'X').
		MtxCV::printImageAsciiSerial(edgeMask, true, false, true);
	}

}