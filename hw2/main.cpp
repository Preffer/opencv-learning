#include <iostream>
#include <boost/format.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace boost;
using namespace cv;

typedef vector<Point> Contour;
typedef pair<Contour, float> ContourRecord;
const int borderS = 10;

int main(int argc, char *argv[]) {
	CommandLineParser cmd(argc, argv,
		"{ 1 |      |       | Set input image }"
		"{ h | help | false | Show this help message }"
	);

	if(cmd.get<bool>("help")){
		cout << "Usage: ./main <input>" << endl;
		cout << "Options:" << endl;
		cmd.printParams();
		return EXIT_SUCCESS;
	}

	string inputPath = cmd.get<string>("1");
	if(inputPath.empty()){
		cout << "Please specific input image" << endl;
		cout << "Usage: ./main <input>" << endl;
		cout << "Options:" << endl;
		cmd.printParams();
		return EXIT_FAILURE;
	}

	Mat src = imread(inputPath);
	if(src.empty()){
		cout << boost::format("Failed to open %1%") % inputPath << endl;
		return EXIT_FAILURE;
	}

	imshow("Source", src);

	Mat step;
	cvtColor(src, step, COLOR_BGR2GRAY);
	imshow("Grey", step);

	dilate(step, step, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	imshow("Dilation", step);

	normalize(step, step, 0, 255, NORM_MINMAX);
	imshow("Normalize", step);

	GaussianBlur(step, step, Size(3, 3), 0);
	imshow("GaussianBlur", step);

	threshold(step, step, 0, 255, THRESH_BINARY | THRESH_OTSU);
	imshow("Threshold", step);

	copyMakeBorder(step, step, borderS, borderS, borderS, borderS, BORDER_CONSTANT, CV_RGB(255, 255, 255));
	imshow("MakeBorder", step);

	vector<Contour> rawContours;
	vector<Vec4i> hierarchy;

	findContours(step, rawContours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	step = Mat::zeros(step.size(), CV_8UC1);
	drawContours(step, rawContours, -1, CV_RGB(255, 255, 255));
	imshow("RawContours", step);

	vector<Contour> contours;
	for(int i = rawContours.size() - 1; i >= 0; i--){
		if(rawContours[i].size() < 5){
			continue;
		}
		if(hierarchy[i][3] >= 0){
			contours.push_back(rawContours[i]);
		}
	}

	ContourRecord maxCell(contours[0], contourArea(contours[0]));
	ContourRecord minCell(maxCell);
	float totalArea = 0;
	for(Contour& contour : contours){
		float area = contourArea(contour);
		if(area > maxCell.second){
			maxCell = ContourRecord(contour, area);
		}
		if(area < minCell.second){
			minCell = ContourRecord(contour, area);
		}
		totalArea += area;
	}
	RotatedRect maxBox = fitEllipse(maxCell.first);
	RotatedRect minBox = fitEllipse(minCell.first);

	step = Mat::zeros(step.size(), CV_8UC1);
	drawContours(step, contours, -1, CV_RGB(255, 255, 255));
	imshow("FilteredContours", step);

	cout << boost::format("Total:\n\t%1% cells") % contours.size() << endl;
	cout << boost::format("Max cell:\n\tArea: %1%\n\tArcLength: %2%\n\tOrientation: %3%\n\tCenter: %4%") % maxCell.second % arcLength(maxCell.first, true) % maxBox.angle % (maxBox.center - Point2f(borderS, borderS)) << endl;
	cout << boost::format("Min cell:\n\tArea: %1%\n\tArcLength: %2%\n\tOrientation: %3%\n\tCenter: %4%") % minCell.second % arcLength(minCell.first, true) % minBox.angle % (minBox.center - Point2f(borderS, borderS)) << endl;
	cout << boost::format("Average:\n\tCell area: %1%") % (totalArea / contours.size()) << endl;

	waitKey();
	return EXIT_SUCCESS;
}
