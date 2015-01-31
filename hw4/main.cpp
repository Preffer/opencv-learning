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

typedef tuple<Mat, int, string> Sample;

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

	try {
		int index = 0;
		for(auto it = directory_iterator(inputDir); it != directory_iterator(); it++){
			if(is_directory(it->path())){
				string name = it->path().leaf().string();
				for(auto img = directory_iterator(it->path().string()); img != directory_iterator(); img++){
					samples.push_back(Sample(imread(img->path().string(), CV_LOAD_IMAGE_GRAYSCALE), index, name));
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

	if(samples.size() <= numTestCase) {
		cerr << boost::format("No enough photos, you request %1% test cases, but you only give %2% photos") % numTestCase % samples.size() << endl;
		return EXIT_FAILURE;
	}

	shuffle(samples.begin(), samples.end(), default_random_engine());

	vector<Mat> images, testImages;
	vector<int> labels, testLabels;
	vector<string> names, testNames;

	for(int i = samples.size() - 1; i >= 0; i--){
		if(i < 10){
			testImages.push_back(get<0>(samples[i]));
			testLabels.push_back(get<1>(samples[i]));
			testNames.push_back(get<2>(samples[i]));
		} else{
			images.push_back(get<0>(samples[i]));
			labels.push_back(get<1>(samples[i]));
			names.push_back(get<2>(samples[i]));
		}
	}

	// This here is a full PCA, if you just want to keep
	// 10 principal components (read Eigenfaces), then call
	// the factory method like this:
	//
	//      cv::createEigenFaceRecognizer(10);
	//
	// If you want to create a FaceRecognizer with a
	// confidennce threshold, call it with:
	//
	//      cv::createEigenFaceRecognizer(10, 123.0);
	//
	Ptr<FaceRecognizer> model = createEigenFaceRecognizer(50);
	model->train(images, labels);
	model->save("model.xml");
	cout << "Model has been saved to model.xml" << endl;

	for(uint i = 0; i < numTestCase; i++){
		int predictedLabel;
		double confidence;
		model->predict(testImages[i], predictedLabel, confidence);
		cout << boost::format("Predicted class = %1%, Actual class = %2%, Confidence = %3%") % predictedLabel % testLabels[i] % confidence << endl;
	}

	return EXIT_SUCCESS;
}