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

	// k-means part
	accumulator_set<float, stats<tag::min, tag::max, tag::mean> > initAcc;
	for(ContourRecord& record : candidateRecords){
		initAcc(record.second);
	}
	float smallMean = accumulators::min(initAcc);
	float middleMean = accumulators::mean(initAcc);
	float bigMean = accumulators::max(initAcc);
	float distance = FLT_MAX;

	cout << "Running K-means..." << endl;
	while(true){
		cout << boost::format("SmallMean: %1%, MiddleMean: %2%, bigMean: %3%, distance: %4%") % smallMean % middleMean % bigMean % distance << endl;
		float smallLimit = (smallMean + middleMean) / 2;
		float bigLimit = (middleMean + bigMean) / 2;
		accumulator_set<float, stats<tag::count, tag::mean, tag::variance> > smallAcc;
		accumulator_set<float, stats<tag::count, tag::mean, tag::variance> > middleAcc;
		accumulator_set<float, stats<tag::count, tag::mean, tag::variance> > bigAcc;
		for(ContourRecord& record : candidateRecords){
			if(record.second < smallLimit){
				smallAcc(record.second);
			} else{
				if(record.second > bigLimit){
					bigAcc(record.second);
				} else{
					middleAcc(record.second);
				}
			}
		}

		float _smallMean = accumulators::mean(smallAcc);
		float _middleMean = accumulators::mean(middleAcc);
		float _bigMean = accumulators::mean(bigAcc);
		float _distance = accumulators::variance(smallAcc) * accumulators::count(smallAcc)
						+ accumulators::variance(middleAcc) * accumulators::count(middleAcc)
						+ accumulators::variance(bigAcc) * accumulators::count(bigAcc);
		
		if(distance > _distance){
			smallMean = _smallMean;
			middleMean = _middleMean;
			bigMean = _bigMean;
			distance = _distance;
		} else{
			break;
		}
	}
	cout << boost::format("Complete K-means. Minimum valid cell size: %1%") % smallMean << endl << endl;

	candidateRecords.remove_if([&](ContourRecord& record){
		if(record.second < smallMean){
			return true;
		} else{
			return false;
		}
	});

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
