#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

const string title = "Threshold";

int main(int argc, char *argv[]) {
	CommandLineParser cmd(argc, argv,
		"{ i | input        | input.avi  | Input video }"
		"{ t | threshold    | 128        | Threshold }"
		"{ o | output       | output     | Output video }"
		"{ h | help         | false      | print help message }");
	
	if(cmd.get<bool>("help")){
		cout << "Usage: main [options]" << endl;
		cout << "Available options:" << endl;
		cmd.printParams();
		return EXIT_SUCCESS;
	}

	string inputFile = cmd.get<string>("input");
	int thresholdValue = cmd.get<int>("threshold");
	string outputFile = cmd.get<string>("output");

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

	Mat inputFrame, outputFrame;
	Point position(
		input.get(CV_CAP_PROP_FRAME_WIDTH) * 0.05,
		input.get(CV_CAP_PROP_FRAME_HEIGHT) * 0.95
	);
	Scalar color(255, 255, 255);
	int interval = 1000 / input.get(CV_CAP_PROP_FPS);
	namedWindow(title, CV_WINDOW_AUTOSIZE);

	while(true){
		input >> inputFrame;
		if(inputFrame.empty()){
			break;
		}

		cvtColor(inputFrame, inputFrame, COLOR_BGR2GRAY);
		threshold(inputFrame, outputFrame, thresholdValue, 255, THRESH_BINARY);
		putText(outputFrame, "Huang Yuzhong - 3120102663", position, FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
		imshow(title, outputFrame);
		output << outputFrame;
		waitKey(interval);
	}

	return EXIT_SUCCESS;
}