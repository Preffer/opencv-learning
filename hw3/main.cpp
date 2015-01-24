#include <list>
#include <iostream>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace boost;
using namespace std;
namespace po = program_options;

typedef list<string> FileList;
typedef vector<Point2f> Points;
typedef vector<Points> PointsRecord;

FileList readDir(string& dir);

int main(int argc, char *argv[]) {
	string inputDir, undistortDir;
	int row, col, squareSize;

	po::options_description desc("Options");
	desc.add_options()
		("input,i", po::value<string>(&inputDir), "Input image directory")
		("undistort,u", po::value<string>(&undistortDir), "Undistort image directory")
		("row,r", po::value<int>(&row)->default_value(12), "Rows of the board")
		("col,c", po::value<int>(&col)->default_value(12), "Cows of the board")
		("size,s", po::value<int>(&squareSize)->default_value(50), "Size of the square")
		("help,h", "Show this help info");

	po::variables_map vm;
	try{
		po::store(po::parse_command_line(argc, argv, desc), vm);
		if (vm.count("help")) {
			cout << desc << endl;
			return EXIT_SUCCESS;
		}
		po::notify(vm);
	} catch(po::error& e) {
		cerr << "Error: " << e.what() << endl << endl;
		cout << desc << endl;
		return EXIT_FAILURE;
	}

	Size boardSize(row, col);
	Size imageSize = imread(*readDir(inputDir).begin()).size();
	namedWindow("Image", WINDOW_NORMAL);
	
	PointsRecord imagePoints;

	for(string& fileName : readDir(inputDir)){
		Mat frame = imread(fileName);
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
			imagePoints.push_back(corners);
			imshow("Image", frame);
			waitKey(10);
		}
	}

	vector<Mat> rvecs, tvecs;
	vector<float> reprojErrs;
	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
	cameraMatrix.at <double>(0, 0) = 1.0;
	cout << "OK" << endl;
	Mat distCoeffs = Mat::zeros(8, 1, CV_64F);
	
	cout << "OK" << endl;
	vector < vector < Point3f > >  objectPoints(1);
	objectPoints[0].clear();

	for (int i = 0; i < boardSize.height; i++) {
		for (int j = 0; j < boardSize.width; j++){
			objectPoints[0].push_back(Point3f(float(j * squareSize), float(i * squareSize), 0));
		}		
	}
	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_FIX_PRINCIPAL_POINT | CV_CALIB_FIX_ASPECT_RATIO | CV_CALIB_ZERO_TANGENT_DIST | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5);
	cout << boost::format("Re-projection error reported by calibrateCamera: %1%") % rms << endl;

	bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);
	cout << boost::format("OK? %1%") % ok << endl;
	
	double totalAvgErr;
	
	vector < Point2f > imagePoints2;
	int totalPoints = 0;
	double totalErr = 0, err;
	reprojErrs.resize(objectPoints.size());

	for (int i = 0; i < (int)objectPoints.size(); ++i) {
		projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
		err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);

		int n = (int)objectPoints[i].size();
		reprojErrs[i] = (float)std::sqrt(err * err / n);
		totalErr += err * err;
		totalPoints += n;
	}
	totalAvgErr = sqrt(totalErr / totalPoints);
	cout << boost::format("totalAvgErr: %1%") % totalAvgErr << endl;


	Mat view, rview, map1, map2;
	initUndistortRectifyMap(
		cameraMatrix,
		distCoeffs,
		Mat(),
		getOptimalNewCameraMatrix(
			cameraMatrix,
			distCoeffs,
			imageSize,
			1,
			imageSize,
			0
		),
		imageSize,
		CV_16SC2,
		map1,
		map2
	);

	for(string& fileName : readDir(undistortDir)){
		view = imread(fileName);
		if (view.empty()){
			cerr << boost::format("Failed to read %1%, ignored.") % fileName << endl;
			continue;
		}
		remap(view, rview, map1, map2, INTER_NEAREST);
		imshow("Image", rview);
		waitKey();
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
