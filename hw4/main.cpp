#include <iostream>
#include <algorithm>
#include <boost/timer.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

using namespace std;
using namespace boost;
using namespace filesystem;
using namespace cv;

typedef tuple<Mat, int> Sample;

void logTime(const string& message);

int main(int argc, char *argv[]) {
	logTime("Launched");

	CommandLineParser cmd(argc, argv,
		"{ 1 |       |       | Photos directory }"
		"{ t | test  | 10    | Test set size }"
		"{ l | limit | 100   | Max samples for each label }"
		"{ m | model | e     | e(Eigenfaces)/f(Fisherfaces)/l(LBPH) }"
		"{ d | dim   | 100   | Dimension of PCA, only for Eigenfaces }"
		"{ h | help  | false | Show this help message }"
	);

	string inputDir = cmd.get<string>("1");
	if(inputDir.empty() || cmd.get<bool>("help")){
		cout << boost::format("Usage: %1% <directory>") % argv[0] << endl;
		cout << "Options:" << endl;
		cmd.printParams();
		return EXIT_FAILURE;
	}

	uint numTestCase = cmd.get<uint>("test");
	int limit = cmd.get<int>("limit");
	char modelName = cmd.get<string>("model").front();
	int dim = cmd.get<int>("dim");
	vector<Sample> samples;
	map<int, string> names;

	try {
		int index = 0;
		for(auto it = directory_iterator(inputDir); it != directory_iterator(); it++){
			if(is_directory(it->path())){
				names[index] = it->path().leaf().string();
				int count = 1;
				for(auto img = directory_iterator(it->path().string()); img != directory_iterator(); img++){
					if(count++ <= limit){
						samples.push_back(Sample(imread(img->path().string(), CV_LOAD_IMAGE_GRAYSCALE), index));
					} else{
						break;
					}
				}
				index++;
			}
		}
	} catch(filesystem_error& ex) {
		cerr << ex.what() << endl;
		exit(EXIT_FAILURE);
	} catch (Exception& ex) {
		cerr << ex.what() << endl;
		exit(EXIT_FAILURE);
	}

	cout << boost::format(
		"Parameters:\n"
		"\tPhotos directory: %1%\n"
		"\tTrain set size: %2%\n"
		"\tTest set size: %3%\n"
		"\tSample labels: %4%\n"
		"\tMax Sample per label: %5%\n"
		"\tModel Use: %6%\n"
	) % inputDir % (samples.size() - numTestCase) % numTestCase % names.size() % limit % modelName;

	if(modelName == 'e'){
		cout << boost::format("\tDimension: %1%\n") % dim << endl;
	} else{
		cout << endl;
	}

	if(names.size() <= numTestCase) {
		cerr << boost::format("No enough photos, you request %1% test cases, but the photos only have %2% labels") % numTestCase % names.size() << endl;
		exit(EXIT_FAILURE);
	}

	shuffle(samples.begin(), samples.end(), default_random_engine(time(NULL)));

	vector<Mat> images, testImages;
	vector<int> labels, testLabels;

	uint sampleSize = samples.size();
	for(uint i = 0; i < sampleSize; i++){
		if(i < numTestCase){
			testImages.push_back(get<0>(samples[i]));
			testLabels.push_back(get<1>(samples[i]));
		} else{
			images.push_back(get<0>(samples[i]));
			labels.push_back(get<1>(samples[i]));
		}
	}

	Ptr<FaceRecognizer> model;
	switch(modelName){
		case 'e':
			model = createEigenFaceRecognizer(dim);
			break;
		case 'f':
			model = createFisherFaceRecognizer();
			break;
		case 'l':
			model = createLBPHFaceRecognizer();
			break;
		default:
			cerr << "Unknown model" << endl;
			exit(EXIT_FAILURE);
	}

	logTime("Before training");
	model->train(images, labels);
	logTime("After training");

	int correct = 0;
	logTime("Before predication");
	for(uint i = 0; i < numTestCase; i++){
		int predicate;
		double confidence;
		model->predict(testImages[i], predicate, confidence);
		cout << boost::format("Predicate: %1%, Actual: %2%, Confidence: %3%") % names[predicate] % names[testLabels[i]] % confidence << endl;

		if(predicate == testLabels[i]){
			correct++;
		}

		string text = (boost::format("%1%/%2%") % names[predicate] % names[testLabels[i]]).str();
		Point pos(testImages[i].size().width * 0.02, testImages[i].size().height * 0.98);
		putText(testImages[i], text, pos,  FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 255, 255));

		string title = "Test Case - " + to_string(i);
		namedWindow(title, WINDOW_NORMAL);
		imshow(title, testImages[i]);
	}
	cout << "Accuracy: " << (100.0 * correct / numTestCase) << '%' << endl;
	logTime("Finished");

	waitKey();
	return EXIT_SUCCESS;
}

void logTime(const string& message) {
	static timer t;
	cout << boost::format("[%1%] %2%") % t.elapsed() % message << endl;
}
