#include <iostream>
#include <boost/format.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace boost;
using namespace cv;

typedef vector<Point> Contour;
typedef pair<Contour, float> ContourRecord;

const int borderW = 10;

int main(int argc, char **argv) {
	Mat src = imread(argv[1]);
	namedWindow("Source", CV_WINDOW_AUTOSIZE);
	imshow("Source", src);
	
	Mat step;
	cvtColor(src, step, COLOR_BGR2GRAY);
	namedWindow("Grey", CV_WINDOW_AUTOSIZE);
	imshow("Grey", step);

	dilate(step, step, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	namedWindow("Dilation", CV_WINDOW_AUTOSIZE);
	imshow("Dilation", step);

	normalize(step, step, 0, 255, NORM_MINMAX);
	namedWindow("Normalize", CV_WINDOW_AUTOSIZE);
	imshow("Normalize", step);

	GaussianBlur(step, step, Size(3, 3), 0);
	namedWindow("GaussianBlur", CV_WINDOW_AUTOSIZE);
	imshow("GaussianBlur", step);

	threshold(step, step, 0, 255, THRESH_BINARY | THRESH_OTSU);
	namedWindow("Threshold", CV_WINDOW_AUTOSIZE);
	imshow("Threshold", step);

	copyMakeBorder(step, step, borderW, borderW, borderW, borderW, BORDER_CONSTANT, CV_RGB(255, 255, 255));
	namedWindow("MakeBorder", CV_WINDOW_AUTOSIZE);
	imshow("MakeBorder", step);

	vector<Contour> rawContours;
	vector<Vec4i> hierarchy;

	findContours(step, rawContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	step = Mat::zeros(step.size(), CV_8UC1);
	drawContours(step, rawContours, -1, CV_RGB(255, 255, 255));
	namedWindow("RawContours", CV_WINDOW_AUTOSIZE);
	imshow("RawContours", step);

	vector<Contour> contours;
	ContourRecord maxCell(rawContours[0], contourArea(rawContours[10]));
	ContourRecord minCell(rawContours[0], contourArea(rawContours[10]));
	float totalArea = 0;
	for(int i = rawContours.size() - 1; i >= 0; i--){
		if(rawContours[i].size() < 5){
			continue;
		}
		if(hierarchy[i][3] == 0){
			contours.push_back(rawContours[i]);
			float area = contourArea(rawContours[i]);
			if(area > maxCell.second){
				maxCell = ContourRecord(rawContours[i], area);
			}
			if(area < minCell.second){
				minCell = ContourRecord(rawContours[i], area);
			}
			totalArea += area;
		}
	}

	step = Mat::zeros(step.size(), CV_8UC1);
	drawContours(step, contours, -1, CV_RGB(255, 255, 255));
	namedWindow("TweakContours", CV_WINDOW_AUTOSIZE);
	imshow("TweakContours", step);

	cout << boost::format("%1% cells in total") % contours.size() << endl;
	RotatedRect maxBox = fitEllipse(maxCell.first);
	RotatedRect minBox = fitEllipse(minCell.first);
	cout << boost::format("Max cell:\nArea: %1%, ArcLength: %2%, Orientation: %3%, Center: %4%") % maxCell.second % arcLength(maxCell.first, true) % maxBox.angle % maxBox.center << endl;
	cout << boost::format("Min cell:\nArea: %1%, ArcLength: %2%, Orientation: %3%, Center: %4%") % minCell.second % arcLength(minCell.first, true) % minBox.angle % minBox.center << endl;

	cout << boost::format("Average area: %1%") % (totalArea / contours.size()) << endl;
	cvWaitKey();

	return EXIT_SUCCESS;
}
