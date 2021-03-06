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
void heightChanged(int pos, void* data);
void doTransform(int pos, void* data, bool save = false);

int main(int argc, char *argv[]) {
	string inputDir, transformFile, xmlFile;
	int row, col, squareSize;
	uint displayAmount;

	po::options_description desc("Options");
	desc.add_options()
		("image,i", po::value<string>(&inputDir), "Images directory")
		("transform,t", po::value<string>(&transformFile), "Image for perspective transformation")
		("xml,x", po::value<string>(&xmlFile)->default_value("Calibration.xml"), "Xml file that contain calibration matrix")
		("row,r", po::value<int>(&row)->default_value(12), "Rows of the board")
		("col,c", po::value<int>(&col)->default_value(12), "Cows of the board")
		("size,s", po::value<int>(&squareSize)->default_value(50), "Size of the square")
		("display,d", po::value<uint>(&displayAmount)->default_value(2), "Amount of images to display")
		("help,h", "Show this help info");

	po::variables_map vm;
	try{
		po::store(po::parse_command_line(argc, argv, desc), vm);
		if (vm.count("help")) {
			cout << desc << endl;
			return EXIT_SUCCESS;
		}
		if(!vm.count("image")){
			cout << "You need to specific input images directory" << endl;
			return EXIT_FAILURE;
		}
		if(!vm.count("transform")){
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
	cout << boost::format(
		"Parameters:\n"
		"\tImages directory: %1%\n"
		"\tTransform image: %2%\n"
		"\tXml file: %3%\n"
		"\tChessboard size: %4%\n"
		"\tSquare size: %5%\n"
		"\tDisplay: %6% images\n"
	) % inputDir % transformFile % xmlFile % boardSize % squareSize % displayAmount << endl;

	Files inputFiles = readDir(inputDir);
	Files _inputFiles = inputFiles;

	if(inputFiles.size() < displayAmount){
		cerr << boost::format("No enough images to calibrate, you request to display is %1% images, but you only give %2% images") % displayAmount % inputFiles.size() << endl;
		return EXIT_FAILURE;
	}
	Size imageSize = imread(inputFiles.back()).size();

	Mat transform;
	transform = imread(transformFile);
	if(transform.empty()){
		cerr << boost::format("Failed to read %1% for perspective transformation") % transformFile << endl;
		return EXIT_FAILURE;
	}

	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
	Mat distCoeffs = Mat::zeros(8, 1, CV_64F);
	//try to read from xmlFile
	{
		FileStorage fs(xmlFile, FileStorage::READ);
		fs["CameraMatrix"] >> cameraMatrix;
		fs["DistCoeffs"] >> distCoeffs;
	}

	if(cameraMatrix.empty() || distCoeffs.empty()){
		cout << "Hasn't present xml file to load matrix, calibrating camera from scratch..." << endl;
		ImagePoints imagePoints;

		//inline first few iteration to read image and show them
		thread_group showGroup;
		for(uint i = 0; i < displayAmount; i++){
			string fileName = inputFiles.back();
			inputFiles.pop_back();

			showGroup.create_thread([&, fileName]{
				Mat frame = imread(fileName);
				if(frame.empty()){
					cerr << boost::format("Failed to read %1%, not a vaild image, Abort.") % fileName << endl;
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
					cerr << boost::format("Failed to find chessboard in %1%, Abort. The parameters may not proper.") % fileName << endl;
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
						continue;
					}

					Points corners;
					if(findChessboardCorners(frame, boardSize, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE)){
						cornerSubPix(frame, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.1));
						imagePoints.push_back(corners);
					} else{
						cerr << boost::format("Failed to find chessboard in %1%, ignored.") % fileName << endl;
					}
				}

				return EXIT_SUCCESS;
			});
		}
		findGroup.join_all();

		//do calibration
		ObjectPoints objectPoints(1);
		for (int i = 0; i < boardSize.height; i++) {
			for (int j = 0; j < boardSize.width; j++){
				objectPoints[0].push_back(Point3f(float(j * squareSize), float(i * squareSize), 0));
			}
		}
		objectPoints.resize(imagePoints.size(), objectPoints[0]);

		vector<Mat> rvecs, tvecs;
		calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_FIX_PRINCIPAL_POINT);

		if(checkRange(cameraMatrix) && checkRange(distCoeffs)){
			cout << "相机内参: " << endl;
			cout << "\tX方向焦距: " << cameraMatrix.at<double>(0, 0) << endl;
			cout << "\tY方向焦距: " << cameraMatrix.at<double>(1, 1) << endl;
			cout << "\t投影原点X坐标: " << cameraMatrix.at<double>(0, 2) << endl;
			cout << "\t投影原点Y坐标: " << cameraMatrix.at<double>(1, 2) << endl;
			cout << "畸变参数: " << endl;
			cout << "\t径向畸变k1: " << distCoeffs.at<double>(0, 0) << endl;
			cout << "\t径向畸变k2: " << distCoeffs.at<double>(0, 1) << endl;
			cout << "\t切向畸变p1: " << distCoeffs.at<double>(0, 2) << endl;
			cout << "\t切向畸变p2: " << distCoeffs.at<double>(0, 3) << endl;
			cout << "\t径向畸变k3: " << distCoeffs.at<double>(0, 4) << endl;
		} else{
			cerr << "Calibration give invalid output, failed." << endl;
			return EXIT_FAILURE;
		}

		FileStorage fs(xmlFile, FileStorage::WRITE);
		fs << "CameraMatrix" << cameraMatrix;
		fs << "DistCoeffs" << distCoeffs;
		cout << "Matrix saved to " << xmlFile << endl;
	} else{
		cout << "Successfully load matrix from " << xmlFile << endl;
	}

	//show undistorted images
	Mat raw, undistorted, map1, map2;
	Mat newCameraMatrix = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1);
	initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), newCameraMatrix, imageSize, CV_16SC2, map1, map2);

	inputFiles = _inputFiles;
	for(uint i = 0; i < displayAmount; i++){
		string fileName = inputFiles.back();
		inputFiles.pop_back();

		raw = imread(fileName);
		remap(raw, undistorted, map1, map2, INTER_NEAREST);
		
		string title = "Undistorted - " + fileName;
		namedWindow(title, WINDOW_NORMAL);
		imshow(title, undistorted);
	}
	waitKey(1);

	//show perspective
	int height = 50;
	namedWindow("Perspective", WINDOW_NORMAL);
	createTrackbar("Height", "Perspective", &height, 100, heightChanged, &transform);
	doTransform(height, &transform);

	cout << endl << boost::format("Interactive transforming %1%...") % transformFile << endl;
	cout << "You can press 'q' to quit, and any other key to save an snapshot." << endl;
	while(true){
		char keycode = waitKey();
		if(keycode == 'q' || keycode == 'Q'){
			break;
		} else{
			doTransform(height, &transform, true);
		}
	}

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
		exit(EXIT_FAILURE);
	}

	return files;
}

void heightChanged(int pos, void* data){
	doTransform(pos, data);
}

void doTransform(int pos, void* data, bool save) {
	Mat& transform = *(Mat*)(data);
	Mat perspective;

	Size transSize = transform.size() - Size(1, 1);
	vector<Point2f> from {
		Point2f(0, 0),
		Point2f(transSize.width, 0),
		Point2f(0, transSize.height),
		Point2f(transSize.width, transSize.height)
	};
	//uniform transform, no matter pos > 50 or pos < 50
	vector<Point2f> to {
		Point2f(transSize.width * (50 - pos) / 100.0, 0),
		Point2f(transSize.width * (1 - (50 - pos)/100.0), 0),
		Point2f(transSize.width * (pos - 50) / 100.0, transSize.height),
		Point2f(transSize.width * (1 - (pos - 50)/100.0), transSize.height)
	};

	Mat trans = getPerspectiveTransform(from, to);
	warpPerspective(transform, perspective, trans, transform.size());

	if(save){
		string fileName = (boost::format("Height_%1%.png") % pos).str();
		imwrite(fileName, perspective);
		cout << "Saved: " << fileName << endl;
	} else{
		imshow("Perspective", perspective);
	}
}