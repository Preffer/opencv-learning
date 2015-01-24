#include <list>
#include <iostream>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace boost;
using namespace std;

typedef list<string> FileList;
typedef vector<Point2f> Points;
typedef vector<Points> TrackPoints;

FileList readDir(string& dir);

int main(int argc, char *argv[]) {
	CommandLineParser cmd(argc, argv,
		"{ i | input  |       | Input image directory }"
		"{ r | row    |       | Rows of the board}"
		"{ c | cow    |       | Cows of the cow}"
		"{ h | help   | false | Show this help message }"
	);

	if(cmd.get<bool>("help")){
		cout << "Options:" << endl;
		cmd.printParams();
		return EXIT_SUCCESS;
	}

	string inputDir = cmd.get<string>("input");
	if(inputDir.empty()){
		cout << "Please specific input image directory" << endl;
		cout << "Options:" << endl;
		cmd.printParams();
		return EXIT_FAILURE;
	}

	Size boardSize(cmd.get<int>("row"), cmd.get<int>("cow"));
	namedWindow("Image", WINDOW_NORMAL);

	for(string& fileName : readDir(inputDir)){
		Mat frame = imread(fileName, CV_LOAD_IMAGE_COLOR);
		if(frame.empty()){
			cerr << boost::format("Failed to read %1%, ignored.") % fileName << endl;
			continue;
		}
		
		Points corners;
		bool found = findChessboardCorners(frame, boardSize, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);

		if(found){
			Mat greyFrame;
			cvtColor(frame, greyFrame, COLOR_BGR2GRAY);
			cornerSubPix(greyFrame, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.1));
			drawChessboardCorners(frame, boardSize, Mat(corners), true);
			imshow("Image", frame);
			waitKey();
		}
	}

	return EXIT_SUCCESS;
}

FileList readDir(string& dir){
	filesystem::path p(dir);
	FileList fileList;

	try {
		for_each(filesystem::directory_iterator(p), filesystem::directory_iterator(), [&](auto& fd){
			fileList.push_back(fd.path().string());
		});
	} catch(const filesystem::filesystem_error& ex) {
		cerr << ex.what() << endl;
	}

	return fileList;
}
