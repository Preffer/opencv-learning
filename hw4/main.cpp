#include <ctime>
#include <iostream>
#include <algorithm>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

using namespace std;
using namespace boost;
using namespace filesystem;
using namespace cv;

typedef tuple<Mat, int> Sample;

int main(int argc, char *argv[]) {
	CommandLineParser cmd(argc, argv,
		"{ 1 |      |       | Photos directory }"
		"{ t | test | 10    | Number of test case }"
		"{ h | help | false | Show this help message }"
	);

	string inputDir = cmd.get<string>("1");
	if(inputDir.empty() || cmd.get<bool>("help")){
		cout << boost::format("Usage: %1% <directory>") % argv[0] << endl;
		cout << "Options:" << endl;
		cmd.printParams();
		return EXIT_FAILURE;
	}

	uint numTestCase = cmd.get<int>("test");
	vector<Sample> samples;
	map<int, string> names;

	try {
		int index = 0;
		for(auto it = directory_iterator(inputDir); it != directory_iterator(); it++){
			if(is_directory(it->path())){
				names[index] = it->path().leaf().string();
				for(auto img = directory_iterator(it->path().string()); img != directory_iterator(); img++){
					samples.push_back(Sample(imread(img->path().string(), CV_LOAD_IMAGE_GRAYSCALE), index));
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

	if(names.size() <= numTestCase) {
		cerr << boost::format("No enough photos, you request %1% test cases, but the photos only have %2% labels") % numTestCase % names.size() << endl;
		return EXIT_FAILURE;
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

	//since the performance is enough, set a high dimension
	Ptr<FaceRecognizer> model = createEigenFaceRecognizer(100);
	model->train(images, labels);

	for(uint i = 0; i < numTestCase; i++){
		int predicate;
		double confidence;
		model->predict(testImages[i], predicate, confidence);
		cout << boost::format("Predicate: %1%, Actual: %2%, Confidence: %3%") % names[predicate] % names[testLabels[i]] % confidence << endl;

		string text = (boost::format("%1%/%2%") % names[predicate] % names[testLabels[i]]).str();
		Point pos(testImages[i].size().width * 0.02, testImages[i].size().height * 0.98);
		putText(testImages[i], text, pos,  FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));

		string title = "Test Case - " + to_string(i);
		namedWindow(title, WINDOW_NORMAL);
		imshow(title, testImages[i]);
	}

	waitKey();
	return EXIT_SUCCESS;
}
