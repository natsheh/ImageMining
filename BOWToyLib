/*
 * File:   BOWToyLib.cpp
 * Author: Andrea Moio
 *
 * Created on 29 Novembre 2013, 12.56
 *
 * Simple 2-class SVM classifier based
 * on Bag of Visual Words Model
 *
 *
 */



#include "BOWToyLib.h"
#include <iostream>
using namespace cv;

std::string _CLASS_LABELS[1000];

//read an ascii file containing a list of absolute pathnames
std::vector<string> read_file_list(const string file_list) {
	string f;
	string dirname;
	std::vector<string> files;
	std::ifstream in(file_list.c_str());

	if (not in.is_open()) {
		std::cerr << "Cannot read: " << file_list << std::endl;
		return std::vector<string>();
	}

	while (not in.eof()) {
		in >> f;
		files.push_back(f);
	}
	in.close();

	return files;
}



//create a dictionary in BOW Model
Mat create_dictionary(const string file_list) {
	std::cout << "create_dictionary " << std::flush;

	SurfDescriptorExtractor extractor;	//Descriptor Extractor
	SurfFeatureDetector detector(400);	//Feature Point Detector
	Mat v_data, descriptors, dictionary, image;
	std::vector<string> imglist = read_file_list(file_list);
	std::vector<KeyPoint> keypoints;
	BOWKMeansTrainer bow_trainer(200);

	descriptors = Mat(1, extractor.descriptorSize(), extractor.descriptorType());
	//Loop on all the images
	for (int i=0; i<(int)imglist.size(); ++i) {
		//read an image and check that its not corrupted
		image = imread(imglist[i]);
		if (!image.data or image.empty()) {
			std::cout << "(" << i+1 << "/" << imglist.size() << ")INVALID FILE - SKIPPED: " << imglist[i] << std::endl;
			continue;
		}

		//all images are resized
		resize(image, image, Size(256, 256));
		std::cout << "(" << i+1 << "/" << imglist.size() << ")Processing: " << imglist[i] << std::endl;
		imshow(WIN_NAME, image);
		//waitKey(10);

		//extract SURF feature points
		detector.detect(image, keypoints);

		//compute SURF descriptors
		extractor.compute(image, keypoints, descriptors);

		//store sample in v_data
		v_data.push_back(descriptors);
	}

    if (v_data.empty()) {
        dictionary = Mat();

    } else {
        std::cout << "Dictionary creation..." << std::endl;
        bow_trainer.add(v_data);	//add to the BOW generator the train data
        dictionary = bow_trainer.cluster();	//compute dictionary

    }


    return dictionary;
}



//save a dictionary in xml format
bool save_dictionary(Mat dict, const string xmlfile) {
	FileStorage fs(xmlfile, FileStorage::WRITE);
	if (not fs.isOpened()) {
		return false;
	}

	fs << "dictionary" << dict;
	std::cout << "Dictionary saved: " << xmlfile << std::endl;
	fs.release();


	return true;
}



//load a dictionary from an xml file
bool load_dictionary(Mat& dict, const string xmlfile) {
	FileStorage fs(xmlfile, FileStorage::READ);
	if (not fs.isOpened()) {
		return false;
	}

	fs["dictionary"] >> dict;
	std::cout << "Dictionary loaded: " << xmlfile << std::endl;
	fs.release();


	return true;

}



//SVM training set creation
bool prepare_dataset12(BOWImgDescriptorExtractor& bowEx, Ptr<FeatureDetector> detector,
					const string list1, const string list2, Mat& samples, Mat& labels) {

	std::vector<string> imgfiles;
	std::vector<KeyPoint> keypoints;
	Mat descriptor, l0, l1;

	//class labels
	l0 = Mat(1,1, CV_32F);
	l0.at<float>(0,0) = 0;
	l1 = Mat(1,1, CV_32F);
	l1.at<float>(0,0) = 1;


	//loop an all images in dataset (first class)
	imgfiles = read_file_list(list1);
	for (int i=0; i<(int)imgfiles.size()-1; ++i) {
		Mat img = imread(imgfiles[i]);
		if (!img.data or img.empty()) {
			std::cout << "(" << i+1 << "/" << imgfiles.size() << ")INVALID FILE - SKIPPED: " << imgfiles[i] << std::endl;
			continue;
		}

		resize(img, img, Size(256, 256));
		imshow("ImageClassification", img);
		waitKey(10);
		std::cout << "(" << i+1 << "/" << imgfiles.size() << ")Descriptors extraction: " << imgfiles[i] << std::endl;

		//detect feature points
		detector->detect(img, keypoints);

		//compute descriptors according to our BOW dictionary
		bowEx.compute(img, keypoints, descriptor);
		samples.push_back(descriptor);	//store descriptor sample
		labels.push_back(l0);	//store label
	}


	//loop on all images in dataset (class 2)
	imgfiles = read_file_list(list2);
	for (int i=0; i<(int)imgfiles.size()-1; ++i) {
		Mat img = imread(imgfiles[i]);
		if (!img.data or img.empty()) {
			std::cout << "(" << i+1 << "/" << imgfiles.size() << ")INVALID FILE - SKIPPED: " << imgfiles[i] << std::endl;
			continue;
		}

		resize(img, img, Size(256, 256));
		imshow("ImageClassification", img);
		waitKey(10);
		std::cout << "(" << i+1 << "/" << imgfiles.size() << ")Descriptors extraction: " << imgfiles[i] << std::endl;
		detector->detect(img, keypoints);
		bowEx.compute(img, keypoints, descriptor);
		samples.push_back(descriptor);
		labels.push_back(l1);
	}

	std::cout << "Training set creation completed." << std::endl;
	std::cout << "N. samples: " << samples.rows << std::endl;

	return true;
}

bool prepare_dataset(BOWImgDescriptorExtractor& bowEx, Ptr<FeatureDetector> detector,
					 string list1[], Mat& samples, Mat& labels) {

	std::vector<string> imgfiles;
	std::vector<KeyPoint> keypoints;
	Mat descriptor;
	int length_list= sizeof(list1)/sizeof(*list1);
	Mat l0[length_list];



	//loop an all images in dataset
	for (int ind=0;ind<length_list;ind+=1)
    {
    l0[ind] = Mat(1,1, CV_32F);
	l0[ind].at<float>(0,0) = ind;

	imgfiles = read_file_list(list1[ind]);
	for (int i=0; i<(int)imgfiles.size()-1; ++i) {
		Mat img = imread(imgfiles[i]);
		if (!img.data or img.empty()) {
			std::cout << "(" << i+1 << "/" << imgfiles.size() << ")INVALID FILE - SKIPPED: " << imgfiles[i] << std::endl;
			continue;
		}

		resize(img, img, Size(256, 256));
		imshow("ImageClassification", img);
		//waitKey(10);

		//detect feature points
		std::cout << "(" << i+1 << "/" << imgfiles.size() << ") before Descriptors extraction: " << imgfiles[i] << std::endl;

		detector->detect(img, keypoints);
        std::cout << "(" << i+1 << "/" << imgfiles.size() << ")compute Descriptors extraction: " << imgfiles[i] << std::endl;

		//compute descriptors according to our BOW dictionary
		bowEx.compute(img, keypoints, descriptor);

		samples.push_back(descriptor);	//store descriptor sample
		labels.push_back(l0[ind]);	//store label
	}

	}


	std::cout << "Training set creation completed." << std::endl;
	std::cout << "N. samples: " << samples.rows << std::endl;

	return true;
}



//load a pre-prepared dataset from an xmlfile
bool load_dataset(const string filename, Mat& samples, Mat& labels) {
	FileStorage fs(filename, FileStorage::READ);
	if (not fs.isOpened()) {
		return false;
	}

	fs["samples"] >> samples;
	fs["labels"] >> labels;
	fs["class1"] >> _CLASS_LABELS[0];
	fs["class2"] >> _CLASS_LABELS[1];
	std::cout << "Training data loaded: " << filename << std::endl;

	return true;
}



//save a dataset in xml format
bool save_dataset(const string filename, Mat samples, Mat labels) {
	FileStorage fs(filename, FileStorage::WRITE);
	if (not fs.isOpened()) {
		return false;
	}

	fs << "samples" << samples;
	fs << "labels" << labels;
	fs << "class1" << _CLASS_LABELS[0];
	fs << "class2" << _CLASS_LABELS[1];
	fs.release();

	std::cout << "Training data saved: " << filename << std::endl;
	return true;
}



//SVM training function
bool trainSVM(CvSVM& classifier, Mat samples, Mat labels) {
    CvSVMParams param;

    param.svm_type = SVM::C_SVC;
    param.kernel_type = SVM::RBF; //CvSVM::RBF, CvSVM::LINEAR ...
    param.degree = 0; // for poly
    param.gamma = 20; // for poly/rbf/sigmoid
    param.coef0 = 0; // for poly/sigmoid

    param.C = 7; // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
    param.nu = 0.0; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
    param.p = 0.0; // for CV_SVM_EPS_SVR

    param.class_weights = NULL; // for CV_SVM_C_SVC
    param.term_crit.type = CV_TERMCRIT_ITER +CV_TERMCRIT_EPS;
    param.term_crit.max_iter = 1000;
    param.term_crit.epsilon = 1e-6;

    //train classifier
    if (classifier.train(samples,labels, Mat(), Mat(), param)) {
		std::cout << "SVM training completed." << std::endl;
		return true;
	} else {
		std::cout << "SVM training FAILED." << std::endl;
		return false;
	}
}



//SVM test function
void testSVM(CvSVM& classifier, BOWImgDescriptorExtractor bowEx,
					Ptr<FeatureDetector> detector, const string file_list) {

	Mat descriptor;
	std::vector<KeyPoint> keypoints;
	std::vector<string> imgfiles = read_file_list(file_list);

	//loop on all the images in the test set
	for (int i=0; i<(int)imgfiles.size(); ++i) {
		Mat img = imread(imgfiles[i]);
		if (img.data and !img.empty()) {
			resize(img, img, Size(256, 256));
			std::cout << "(" << i+1 << "/" << imgfiles.size() << ")Predicting: " << imgfiles[i] << std::endl;

			//find SURF feature points
			detector->detect(img, keypoints);
			//compute BOW descriptors
			bowEx.compute(img, keypoints, descriptor);

			//SVM prediction
			int prediction = (int)classifier.predict(descriptor);
			std::cout << "SVM Prediction: " << _CLASS_LABELS[prediction] << std::endl;
			imshow("ImageClassification", img);
			char k = waitKey(0);
			if (k == 'q') {
				break;
			}

		} else {
			std::cout << "(" << i+1 << "/" << imgfiles.size() << ")INVALID FILE - SKIPPED: " << imgfiles[i] << std::endl;
        }
	}

	return;
}
