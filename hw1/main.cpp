#include <iostream>
#include <boost/format.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "i18nText.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
	CommandLineParser cmd(argc, argv,
		"{ 1 |      | input.avi        | Input video }"
		"{ 2 |      | 128              | Threshold }"
		"{ 3 |      | output           | Output video }"
		"{ h | help | false            | print help message }"
		"{ f | font | wqy-microhei.ttc | font used to display text }");
	
	if(cmd.get<bool>("help")){
		cout << "Options:" << endl;
		cmd.printParams();
		return EXIT_SUCCESS;
	}

	string title = "Threshold";
	string inputFile = cmd.get<string>("1");
	int thresholdValue = cmd.get<int>("2");
	string outputFile = cmd.get<string>("3");
	string font = cmd.get<string>("font");

	VideoCapture input(inputFile);

	if(!input.isOpened()){
		cerr << "Failed to open " << inputFile << endl;
		return EXIT_FAILURE;
	}

	if(outputFile.rfind('.') == string::npos){
		outputFile += inputFile.substr(inputFile.rfind('.'));
	}
	VideoWriter output(
		outputFile,
		input.get(CV_CAP_PROP_FOURCC),
		input.get(CV_CAP_PROP_FPS) * 2,
		Size(input.get(CV_CAP_PROP_FRAME_WIDTH ), input.get(CV_CAP_PROP_FRAME_HEIGHT)),
		false
	);
	if(!output.isOpened()){
		cerr << "Failed to open " << outputFile << endl;
		return EXIT_FAILURE;
	}

	i18nText i18n(font, 30);
	int interval = 1000 / input.get(CV_CAP_PROP_FPS);
	Point position(
		input.get(CV_CAP_PROP_FRAME_WIDTH) * 0.05,
		input.get(CV_CAP_PROP_FRAME_HEIGHT) * 0.95
	);

	cout << boost::format("Converting %1% to %2% with threshold = %3% ...") % inputFile % outputFile % thresholdValue << endl;

	Mat frame;
	namedWindow(title, CV_WINDOW_AUTOSIZE);

	while(true){
		input >> frame;
		if(frame.empty()){
			break;
		}
		i18n.putText(frame, L"黄羽众/3120102663", position, CV_RGB(255, 255, 255));
		cvtColor(frame, frame, COLOR_BGR2GRAY);
		threshold(frame, frame, thresholdValue, 255, THRESH_BINARY);
		imshow(title, frame);
		output << frame;
		waitKey(interval);
	}

	return EXIT_SUCCESS;
}
