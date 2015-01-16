#include <iostream>
#include <list>
#include <boost/format.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace boost;
using namespace boost::accumulators;
using namespace cv;

typedef vector<Point> Contour;
typedef pair<Contour, float> ContourScore;

const int borderW = 10;

int main(int argc, char **argv) {
	Mat src = imread(argv[1]);
	namedWindow("Source", CV_WINDOW_AUTOSIZE);
	imshow("Source", src);
	
	Mat step;
	cvtColor(src, step, COLOR_BGR2GRAY);
	namedWindow("Grey", CV_WINDOW_AUTOSIZE);
	imshow("Grey", step);

	normalize(step, step, 0, 255, NORM_MINMAX);
	namedWindow("Normalize", CV_WINDOW_AUTOSIZE);
	imshow("Normalize", step);

	GaussianBlur(step, step, Size(3, 3), 0);
	namedWindow("GaussianBlur", CV_WINDOW_AUTOSIZE);
	imshow("GaussianBlur", step);

	threshold(step, step, 128, 255, THRESH_BINARY);
	namedWindow("Threshold", CV_WINDOW_AUTOSIZE);
	imshow("Threshold", step);

	copyMakeBorder(step, step, borderW, borderW, borderW, borderW, BORDER_CONSTANT, CV_RGB(255, 255, 255));
	namedWindow("MakeBorder", CV_WINDOW_AUTOSIZE);
	imshow("MakeBorder", step);

	vector<Contour> contours;
	findContours(step, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	contours.pop_back();
	drawContours(step, contours, -1, CV_RGB(255, 255, 255));
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", step);
	cout << contours.size() << endl;

	list<ContourScore> scores;
	accumulator_set<float, stats<tag::mean> > roughAcc;
	for(Contour& contour : contours){
		if(contour.size() < 5){
			continue;
		} else{
			float score = contourArea(contour);
			scores.push_back(ContourScore(contour, score));
			roughAcc(score);
		}
	}

	float mean = accumulators::mean(roughAcc);
	cout << "Mean: " << mean << endl;
	float minLimit = mean / 100;
	float maxLimit = mean * 10;

	scores.remove_if([&](ContourScore score) {
		if(score.second < minLimit || score.second > maxLimit){
			return true;
		} else{
			return false;
		}
	});

	accumulator_set<float, stats<tag::variance> > fineAcc;
	for(ContourScore& score : scores){
		fineAcc(score.second);
	}

	mean = accumulators::mean(fineAcc);
	int vari = sqrt(variance(fineAcc)) * 2;
	cout << "Mean: " << mean << endl;
	cout << "Variance: " << vari << endl;
	scores.remove_if([&](ContourScore score) {
		return abs(score.second - mean) > vari;
	});

	Mat result(step.size(), CV_8UC1);
	cout << scores.size() << endl;
	for(ContourScore& score : scores){
		RotatedRect box = fitEllipse(score.first);

		ellipse(result, box.center, box.size, box.angle, 0, 360, CV_RGB(0, 0, 255), 1, CV_AA);

		//cout << boost::format("中心位置：x=%1%, y=%2%") % box.center.x % box.center.y << endl;
		//cout << boost::format("半径或长短轴长度：w=%1%, h=%2%") % box.size.width % box.size.height << endl << endl;
	}

	namedWindow("Result", CV_WINDOW_AUTOSIZE);
	imshow("Result", result);

	cvWaitKey();

	return EXIT_SUCCESS;
}
