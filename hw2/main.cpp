#include <iostream>
#include <list>
#include <boost/format.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
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
	for(int i = rawContours.size() - 1; i >= 0; i--){
		if(rawContours[i].size() < 5){
			continue;
		}
		if(hierarchy[i][3] >= 0){
			candidateRecords.push_back(ContourRecord(rawContours[i], contourArea(rawContours[i])));
		}
	}

	// k-means
	// loops to filter small objects
	while(true){
		accumulator_set<float, stats<tag::min, tag::max, tag::mean, tag::variance> > globalAcc;
		for(ContourRecord& record : candidateRecords){
			globalAcc(record.second);
		}
		float mean = accumulators::mean(globalAcc);
		float smallMean = accumulators::min(globalAcc);
		float bigMean = accumulators::max(globalAcc);
		float deviation = sqrt(accumulators::variance(globalAcc));
		float smallDeviation = INT_MAX;
		float bigDeviation = INT_MAX;
		float distance = INT_MAX;

		// loop to find k-means
		while(true){
			cout << boost::format("Mean: %1%, smallMean: %2%, bigMean: %3%, deviation: %4%, distance: %5%") % mean % smallMean % bigMean % deviation % distance << endl;
			float critical = (smallMean + bigMean) / 2;
			accumulator_set<float, stats<tag::count, tag::mean, tag::variance> > smallAcc;
			accumulator_set<float, stats<tag::count, tag::mean, tag::variance> > bigAcc;
			for(ContourRecord& record : candidateRecords){
				if(record.second < critical){
					smallAcc(record.second);
				} else{
					bigAcc(record.second);
				}
			}

			float _smallMean = accumulators::mean(smallAcc);
			float _bigMean = accumulators::mean(bigAcc);
			float _distance = accumulators::variance(smallAcc) * accumulators::count(smallAcc) + accumulators::variance(bigAcc) * accumulators::count(bigAcc);

			if(distance > _distance){
				smallMean = _smallMean;
				bigMean = _bigMean;
				distance = _distance;
			} else{
				smallDeviation = sqrt(accumulators::variance(smallAcc));
				bigDeviation = sqrt(accumulators::variance(bigAcc));
				break;
			}
		}

		static float rate = 1;
		float minLimit = smallMean - rate * (smallDeviation < bigDeviation ? smallDeviation : bigDeviation);
		int oldSize = candidateRecords.size();
		candidateRecords.remove_if([&](ContourRecord& record){
			if(record.second < minLimit){
				return true;
			} else{
				return false;
			}
		});
		int newSize = candidateRecords.size();
		cout << boost::format("minLimit: %1%, smallDeviation: %2%, bigDeviation: %3%, oldSize: %4%, newSize: %5%") % minLimit % smallDeviation % bigDeviation % oldSize % newSize << endl;
		if(newSize == oldSize){
			break;
		}
		rate *= 1.2;
	}

	ContourRecord maxCell = *candidateRecords.begin();
	ContourRecord minCell = maxCell;

	vector<Contour> contours;
	accumulator_set<float, stats<tag::mean> > finalAcc;
	for(ContourRecord& record : candidateRecords){
		if(record.second > maxCell.second){
			maxCell = record;
		}
		if(record.second < minCell.second){
			minCell = record;
		}
		contours.push_back(record.first);
		finalAcc(record.second);
	}
	RotatedRect maxBox = fitEllipse(maxCell.first);
	RotatedRect minBox = fitEllipse(minCell.first);

	step = Mat::zeros(step.size(), CV_8UC1);
	drawContours(step, contours, -1, CV_RGB(255, 255, 255));
	imshow("FilteredContours", step);

	cout << boost::format("Total:\n\t%1% cells") % contours.size() << endl;
	cout << boost::format("Max cell:\n\tArea: %1%\n\tArcLength: %2%\n\tOrientation: %3%\n\tCenter: %4%") % maxCell.second % arcLength(maxCell.first, true) % maxBox.angle % (maxBox.center - Point2f(borderS, borderS)) << endl;
	cout << boost::format("Min cell:\n\tArea: %1%\n\tArcLength: %2%\n\tOrientation: %3%\n\tCenter: %4%") % minCell.second % arcLength(minCell.first, true) % minBox.angle % (minBox.center - Point2f(borderS, borderS)) << endl;
	cout << boost::format("Average:\n\tCell area: %1%") % accumulators::mean(finalAcc) << endl;

	waitKey();
	return EXIT_SUCCESS;
}
