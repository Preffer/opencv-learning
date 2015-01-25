#include <iostream>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/thread/thread.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace boost;
using namespace std;
namespace po = program_options;

typedef vector<string> Files;
typedef vector<Point2f> Points;
typedef vector<Points> PointsTrack;

Files readDir(string& dir);

int main(int argc, char *argv[]) {
	string inputDir, perspectFile;
	int row, col, squareSize;

	po::options_description desc("Options");
	desc.add_options()
		("input,i", po::value<string>(&inputDir), "Input images directory")
		("perspect,p", po::value<string>(&perspectFile), "Image for perspective transformation")
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
		cerr << boost::format("Error: %1%") % e.what() << endl << endl;
		cout << desc << endl;
		return EXIT_FAILURE;
	}

	Size boardSize(row, col);
	Files inputFiles = readDir(inputDir);

	if(inputFiles.size() < 2){
		cerr << boost::format("No enough images to calibrate, the minium requirement is 2 images, but you give %1% image") % inputFiles.size() << endl;
		return EXIT_FAILURE;
	}

	Size imageSize;
	PointsTrack imagePoints;
	thread_group showGroup;

	//inline first two iteration to read image info and display them
	for(uint i = 0; i < 2; i++){
		showGroup.create_thread([&, i]{
			Mat frame = imread(inputFiles[i]);
			imageSize = frame.size();
			if(frame.empty()){
				cerr << boost::format("Failed to read %1%, not a vaild image. Abort.") % inputFiles[0] << endl;
				exit(EXIT_FAILURE);
			}

			Points corners;
			if(findChessboardCorners(frame, boardSize, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE)){
				Mat greyFrame;
				cvtColor(frame, greyFrame, COLOR_BGR2GRAY);
				cornerSubPix(greyFrame, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.1));
				drawChessboardCorners(frame, boardSize, Mat(corners), true);
				imagePoints.push_back(corners);
				
				string title = "Detection - " + inputFiles[i];
				namedWindow(title, WINDOW_NORMAL);
				imshow(title, frame);
			} else{
				cerr << boost::format("Failed to find chessboard in %1%, the settings may not proper.") % inputFiles[0] << endl;
				cout << boost::format("Current settings: Chessboard Size = %1%") % boardSize << endl;
				exit(EXIT_FAILURE);
			}

			return EXIT_SUCCESS;
		});
	}
	showGroup.join_all();
	waitKey(1);

	//for the later iteration, finish ASAP
	thread_group findGroup;
	uint concurrency = thread::hardware_concurrency();
	for(uint id = 0; id < concurrency; id++){
		findGroup.create_thread([&, id]{
			for(uint i = 2 + id; i < inputFiles.size(); i += concurrency){
				Mat frame = imread(inputFiles[i]);
				if(frame.empty()){
					cerr << boost::format("Failed to read %1%, ignored.") % inputFiles[i] << endl;
					continue;
				}
				
				Points corners;
				if(findChessboardCorners(frame, boardSize, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE)){
					Mat greyFrame;
					cvtColor(frame, greyFrame, COLOR_BGR2GRAY);
					cornerSubPix(greyFrame, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.1));
					drawChessboardCorners(frame, boardSize, Mat(corners), true);
					imagePoints.push_back(corners);
				} else{
					cout << boost::format("Failed to find chessboard in %1%, ignored.") % inputFiles[i] << endl;
				}
			}

			return EXIT_SUCCESS;
		});
	}

	findGroup.join_all();
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
	
	Points imagePoints2;
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
	view = imread(perspectFile);
	Size undistortImageSize = view.size();
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

	vector<Point2f> before(4);
	vector<Point2f> after(4);
	before[0] = Point2f(0, 0);
	before[1] = Point2f(undistortImageSize.width-1, 0);
	before[2] = Point2f(0, undistortImageSize.height-1);
	before[3] = Point2f(undistortImageSize.width-1, undistortImageSize.height-1);
	after[0] = Point2f(0, 0);
	after[1] = Point2f(undistortImageSize.width-1, 0);
	after[2] = Point2f(undistortImageSize.width * 0.4, undistortImageSize.height-1);
	after[3] = Point2f(undistortImageSize.width * 0.6, undistortImageSize.height-1);
	Mat trans = getPerspectiveTransform(before, after);

	if (view.empty()){
		cerr << boost::format("Failed to read %1%.") % perspectFile << endl;
		return EXIT_FAILURE;
	}
	remap(view, rview, map1, map2, INTER_NEAREST);
	warpPerspective(view, rview, trans, undistortImageSize);
	namedWindow("Image", WINDOW_NORMAL);
	imshow("Image", rview);
	waitKey();

	return EXIT_SUCCESS;
}

Files readDir(string& dir) {
	filesystem::path p(dir);
	Files Files;

	try {
		for_each(filesystem::directory_iterator(p), filesystem::directory_iterator(), [&](auto& fd){
			Files.push_back(fd.path().string());
		});
	} catch(const filesystem::filesystem_error& ex) {
		cerr << ex.what() << endl;
	}

	return Files;
}
