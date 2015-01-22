#include <iostream>
#include <list>
#include <boost/format.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/error_of.hpp>
#include <boost/accumulators/statistics/error_of_mean.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace boost;
using namespace boost::accumulators;
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

	list<ContourRecord> candidateRecords;

	accumulator_set<float, stats<tag::min, tag::max, tag::error_of<tag::mean> > > acc;
	for(int i = rawContours.size() - 1; i >= 0; i--){
		if(rawContours[i].size() < 5){
			continue;
		}
		if(hierarchy[i][3] >= 0){
			float area = contourArea(rawContours[i]);
			candidateRecords.push_back(ContourRecord(rawContours[i], area));
			acc(area);
		}
	}

	float smallMean = accumulators::min(acc);
	float bigMean = accumulators::max(acc);
	float various = INT_MAX;
	float error = accumulators::error_of<tag::mean>(acc);

	while(true){
		cout << boost::format("smallMean: %1%, bigMean: %2%, various: %3%") % smallMean % bigMean % various<< endl;

		float critical = (smallMean + bigMean) / 2;
		auto& smallAcc = *(new accumulator_set<float, stats<tag::mean, tag::error_of<tag::mean> > >);
		auto& bigAcc = *(new accumulator_set<float, stats<tag::mean, tag::error_of<tag::mean> > >);
		for(ContourRecord& record : candidateRecords){
			if(record.second < critical){
				smallAcc(record.second);
			} else{
				bigAcc(record.second);
			}
		}
		float thisSmallMean = accumulators::mean(smallAcc);
		float thisbigMean = accumulators::mean(bigAcc);
		float thisVarious = accumulators::error_of<tag::mean>(smallAcc) + accumulators::error_of<tag::mean>(bigAcc);
		delete &smallAcc;
		delete &bigAcc;
		if(thisVarious >=  various){
			smallMean = thisSmallMean;
			bigMean = thisbigMean;
			various = thisVarious;
			break;
		} else{
			various = thisVarious;
			smallMean = thisSmallMean;
			bigMean = thisbigMean;
		}
	}

	float minLimit = smallMean - 3 * error;
	cout << boost::format("Mean: %1%, Error: %2%, Min: %3%") % accumulators::mean(acc) % error % minLimit << endl;

	accumulator_set<float, stats<tag::mean, tag::error_of<tag::mean> > > tmpAcc;
	candidateRecords.remove_if([&](ContourRecord& record){
		if(record.second < minLimit){
			return true;
		} else{
			tmpAcc(record.second);
			return false;
		}
	});

	minLimit = smallMean - 2.35 * accumulators::error_of<tag::mean>(tmpAcc);
	cout << boost::format("Mean: %1%, Error: %2%, Min: %3%") % accumulators::mean(tmpAcc) % accumulators::error_of<tag::mean>(tmpAcc) % minLimit << endl;

	candidateRecords.remove_if([&](ContourRecord& record){
		if(record.second < minLimit){
			return true;
		} else{
			return false;
		}
	});

	ContourRecord maxCell = *candidateRecords.begin();
	ContourRecord minCell = maxCell;

	vector<Contour> contours;
	accumulator_set<float, stats<tag::mean> > meanAcc;
	for(ContourRecord& record : candidateRecords){
		if(record.second > maxCell.second){
			maxCell = record;
		}
		if(record.second < minCell.second){
			minCell = record;
		}
		contours.push_back(record.first);
		meanAcc(record.second);
	}
	RotatedRect maxBox = fitEllipse(maxCell.first);
	RotatedRect minBox = fitEllipse(minCell.first);

	step = Mat::zeros(step.size(), CV_8UC1);
	drawContours(step, contours, -1, CV_RGB(255, 255, 255));
	imshow("FilteredContours", step);

	cout << boost::format("Total:\n\t%1% cells") % contours.size() << endl;
	cout << boost::format("Max cell:\n\tArea: %1%\n\tArcLength: %2%\n\tOrientation: %3%\n\tCenter: %4%") % maxCell.second % arcLength(maxCell.first, true) % maxBox.angle % (maxBox.center - Point2f(borderS, borderS)) << endl;
	cout << boost::format("Min cell:\n\tArea: %1%\n\tArcLength: %2%\n\tOrientation: %3%\n\tCenter: %4%") % minCell.second % arcLength(minCell.first, true) % minBox.angle % (minBox.center - Point2f(borderS, borderS)) << endl;
	cout << boost::format("Average:\n\tCell area: %1%") % accumulators::mean(meanAcc) << endl;

	waitKey();
	return EXIT_SUCCESS;
}
