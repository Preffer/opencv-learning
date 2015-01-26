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
typedef vector<Points> ImagePoints;
typedef vector<Point3f> Point3fs;
typedef vector<Point3fs> ObjectPoints;

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
		if(!vm.count("input")){
			cout << "You need to specific input images directory" << endl;
			return EXIT_FAILURE;
		}
		if(!vm.count("perspect")){
			cout << "You need to specific image for perspective transformation" << endl;
			return EXIT_FAILURE;
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
	ImagePoints imagePoints;
	thread_group showGroup;

	//inline first two iteration to read image info and display them
	for(uint i = 0; i < 2; i++){
		string fileName = inputFiles.back();
		inputFiles.pop_back();

		showGroup.create_thread([&, fileName]{
			Mat frame = imread(fileName);
			imageSize = frame.size();
			if(frame.empty()){
				cerr << boost::format("Failed to read %1%, not a vaild image. Abort.") % fileName << endl;
				exit(EXIT_FAILURE);
			}

			Mat greyFrame;
			cvtColor(frame, greyFrame, COLOR_BGR2GRAY);
			Points corners;
			if(findChessboardCorners(greyFrame, boardSize, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE)){
				cornerSubPix(greyFrame, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.1));
				drawChessboardCorners(frame, boardSize, Mat(corners), true);
				imagePoints.push_back(corners);
				
				string title = "Detection - " + fileName;
				namedWindow(title, WINDOW_NORMAL);
				imshow(title, frame);
			} else{
				cerr << boost::format("Failed to find chessboard in %1%, the settings may not proper.") % fileName << endl;
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
	mutex mtx;
	for(uint id = 0; id < concurrency; id++){
		findGroup.create_thread([&]{
			while(true){
				mtx.lock();
				if(!inputFiles.size()){
					mtx.unlock();
					break;
				}
				string fileName = inputFiles.back();
				inputFiles.pop_back();
				mtx.unlock();

				Mat frame = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
				if(frame.empty()){
					cerr << boost::format("Failed to read %1%, ignored.") % fileName << endl;
					return EXIT_FAILURE;
				}

				Points corners;
				if(findChessboardCorners(frame, boardSize, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE)){
					cornerSubPix(frame, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.1));
					imagePoints.push_back(corners);
				} else{
					cout << boost::format("Failed to find chessboard in %1%, ignored.") % fileName << endl;
				}
			}

			return EXIT_SUCCESS;
		});
	}
	findGroup.join_all();

	ObjectPoints objectPoints(1);
	for (int i = 0; i < boardSize.height; i++) {
		for (int j = 0; j < boardSize.width; j++){
			objectPoints[0].push_back(Point3f(float(j * squareSize), float(i * squareSize), 0));
		}
	}
	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
	Mat distCoeffs = Mat::zeros(8, 1, CV_64F);
	vector<Mat> rvecs, tvecs;
	calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_FIX_PRINCIPAL_POINT | CV_CALIB_FIX_ASPECT_RATIO | CV_CALIB_ZERO_TANGENT_DIST | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5);

	if(checkRange(cameraMatrix) && checkRange(distCoeffs)){
		cout << "CameraMatrix:\n" << cameraMatrix << endl;
		cout << "DistCoeffs:\n" << distCoeffs << endl;
	} else{
		cerr << "Calibration give invalid output. Calibrate failed." << endl;
		return EXIT_FAILURE;
	}

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
	Files files;

	try {
		for(auto it = filesystem::directory_iterator(p); it != filesystem::directory_iterator(); it++){
			files.push_back(it->path().string());
		}
	} catch(const filesystem::filesystem_error& ex) {
		cerr << ex.what() << endl;
	}

	return files;
}
