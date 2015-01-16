#include <iostream>
#include <boost/format.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace boost;
using namespace cv;

typedef vector<Point> Contour;

int main(int argc, char **argv) {
	Mat src = imread(argv[1]);
	namedWindow("Source", CV_WINDOW_AUTOSIZE);
	imshow("Source", src);
	
	Mat step;
	cvtColor(src, step, COLOR_BGR2GRAY);
	namedWindow("Grey", CV_WINDOW_AUTOSIZE);
	imshow("Grey", step);

	//dilate(step, step, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)));
	//namedWindow("Dilation", CV_WINDOW_AUTOSIZE);
	//imshow("Dilation", step);

	normalize(step, step, 0, 255, NORM_MINMAX);
	namedWindow("Normalize", CV_WINDOW_AUTOSIZE);
	imshow("Normalize", step);

	GaussianBlur(step, step, Size(3, 3), 0);
	namedWindow("GaussianBlur", CV_WINDOW_AUTOSIZE);
	imshow("GaussianBlur", step);

	threshold(step, step, 128, 255, THRESH_BINARY);
	namedWindow("Threshold", CV_WINDOW_AUTOSIZE);
	imshow("Threshold", step);

	vector<Contour> contours;
	findContours(step, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	drawContours(step, contours, -1, CV_RGB(255, 255, 255));
	cout << contours.size() << endl;

	Mat result(src.size(), CV_8UC1);
	uint max = min(src.rows, src.cols) / 8;
	for(Contour& contour : contours){
		if(contour.size() < 5 || contour.size() > max){
			continue;
		} else{
			RotatedRect box = fitEllipse(contour);

			ellipse(result, box.center, box.size, box.angle, 0, 360, CV_RGB(0, 0, 255), 1, CV_AA);

			//cout << boost::format("中心位置：x=%1%, y=%2%") % box.center.x % box.center.y << endl;
			//cout << boost::format("半径或长短轴长度：w=%1%, h=%2%") % box.size.width % box.size.height << endl << endl;
		}
	}

	namedWindow("Result", CV_WINDOW_AUTOSIZE);
	imshow("Result", result);

	cvWaitKey();

	return EXIT_SUCCESS;
}
