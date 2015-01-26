/*
 * File:   SVMDemo.cpp
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
#include <cstdlib>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include <cv.h>       // opencv general include file
#include <ml.h>		  // opencv machine learning include file

using namespace cv; // OpenCV API is in the C++ "cv" namespace

#include <stdio.h>

/******************************************************************************/
// global definitions (for speed and ease of use)

#define NUMBER_OF_TRAINING_SAMPLES 3823
#define ATTRIBUTES_PER_SAMPLE 64
#define NUMBER_OF_TESTING_SAMPLES 1797

#define NUMBER_OF_CLASSES 10

bool svm_demo_create_dict(Mat& dict);
bool svm_demo_load_dict(Mat& dict);
bool svm_demo_save_dict(Mat dict);
bool svm_demo_prepare_dataset(BOWImgDescriptorExtractor& bowEx,
            Ptr<FeatureDetector> detector, Mat& samples, Mat& labels,int N_classes);
bool svm_demo_save_dataset(Mat samples, Mat labels);
bool svm_demo_load_dataset(Mat &samples, Mat &labels);
void svm_demo_test(CvSVM& classifier,
    BOWImgDescriptorExtractor bowEx, Ptr<FeatureDetector> detector);

void Rtree_demo_test(CvRTrees& classifier,
                     BOWImgDescriptorExtractor bowEx, Ptr<FeatureDetector> detector);


template <typename T>
string NumberToString ( T Number )
  {
     std::ostringstream ss;
     ss << Number;
     return ss.str();
  }

template <typename T>
int StringToNumber( T str )
  {
    std::istringstream S(str);
    int x;
    S >> x;
    return x;
  }
//main
int main(int argc, char **argv) {
	Mat dictionary, samples, labels, descriptor;
	std::vector<string> imgfiles;
	std::vector<KeyPoint> keypoints;
	Ptr<FeatureDetector> detector = FeatureDetector::create("SURF");	 //SURF feature detector
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);//DescriptorMatcher::create("FlannBased");	//feature matcher
	Ptr<DescriptorExtractor> extractor(new SurfFeatureDetector);// = DescriptorExtractor::create("OpponentSURF");	//SURF descriptor extractor
	BOWImgDescriptorExtractor bowEx(extractor, matcher);	//BOW descriptor extractor
	CvSVM classifier; 	// SVM classifier
    CvRTrees RTreeclassifier;
    string selection = "";
    //string N_classes = "";
    string str="";
    bool done = false;
    bool isTrained = false;

    //default labels
    std::cout << "Enter the Number of classes" << std::endl;
    std::cin.clear();
    std::cin >> str;
    //default labels
    int N_classes=StringToNumber(str);
    for(int i=0;i<N_classes;i+=1){
        _CLASS_LABELS[i] = "class #"+NumberToString(i+1);;
    }
    for(int i=0;i<N_classes;i+=1){
        std::cout <<   _CLASS_LABELS[i] << std::endl;
    }

    while (not done) {
        selection = "";
        std::cout << "BOWImageClassification :: Menu: " << std::endl;
        std::cout << "1: Dictionary Creation" << std::endl;
        std::cout << "2: Dataset Creation" << std::endl;
        std::cout << "3: Classifier Training" << std::endl;
        std::cout << "4: Test" << std::endl;
        std::cout << "5: Save Dictionary" << std::endl;
        std::cout << "6: Save Dataset" << std::endl;
        std::cout << "7: Load Dictionary" << std::endl;
        std::cout << "8: Load Dataset" << std::endl;
        std::cout << "9: Quit" << std::endl;
        std::cout << "\n> " << std::flush;
        std::cin.clear();
        std::cin >> selection;


        //initialize a new dictionary
        if (selection == "1") {
            if (svm_demo_create_dict(dictionary)) {
                bowEx.setVocabulary(dictionary);
                std::cout << "New Dictionary created!" << std::endl;
            } else {
                std::cout << "Dictionary creation failed!" << std::endl;
            }


        //prepare new dataset
        } else if (selection == "2") {
            if (svm_demo_prepare_dataset(bowEx, detector, samples, labels,N_classes)) {
                std::cout << "Dataset created!" << std::endl;
            } else {
                std::cout << "Failed to create dateset!" << std::endl;
            }


        //svm training
        } else if (selection == "3") {
            if (dictionary.empty() or labels.empty() or samples.empty()) {
                std::cout << "Dictionary and dataset must be initialized!" << std::endl;
            } else {
                std::cout << "Training..." << std::endl;
                if (trainSVM(classifier, samples, labels)) {
                    isTrained = true;
                    std::cout << "Classifier trained!" << std::endl;
                } else {
                    std::cout << "Failed to train classifier!" << std::endl;
                }
            }


        //test
        } else if (selection == "4") {
            if (isTrained) {
                svm_demo_test(classifier, bowEx, detector);
                std::cout << "ImageClassification Test: DONE" << std::endl;
            } else {
                std::cout << "Classificator must be trained!" << std::endl;
            }


        //save dictionary (xml file)
        } else if (selection == "5") {
            if (svm_demo_save_dict(dictionary)) {
                std::cout << "Dictionary saved!" << std::endl;
            } else {
                std::cout << "Failed to save dictionary!" << std::endl;
            }


        //save dataset (xml file)
        } else if (selection == "6") {
            if (svm_demo_save_dataset(samples, labels)) {
                std::cout << "Dataset file saved!" << std::endl;
            } else {
                std::cout << "Failed to save dataset!" << std::endl;
            }


        //load dictionary (xml file)
        } else if (selection == "7") {
            if (svm_demo_load_dict(dictionary)) {
                bowEx.setVocabulary(dictionary);
                std::cout << "Dictionary loaded!" << std::endl;
            } else {
                std::cout << "Failed to load dictionary!" << std::endl;
            }


        //load dataset (xml file)
        } else if (selection == "8") {
            if (svm_demo_load_dataset(samples, labels)) {
                std::cout << "Dataset loaded!" << std::endl;
            } else {
                std::cout << "Failed to load dataset!" << std::endl;
            }


        //quit
        } else if (selection == "9") {

            RTreeclassifier=trainRtree( samples,  labels, N_classes);
            Rtree_demo_test(RTreeclassifier,bowEx,detector);

        //default
        } else {
            std::cout << "Invalid Selection!" << std::endl;
        }


        std::cout << "\n\n\n\n" << std::endl;
    }

	return 0;
}



bool svm_demo_create_dict(Mat& dict) {
    string filelist;

    std::cout << "File List: " << std::flush;
    std::cin >> filelist;

    dict = create_dictionary(filelist);
    return (dict.empty() == false);
}



bool svm_demo_load_dict(Mat& dict) {
    string xmlfile;

    std::cout << "Dictionary File (*.xml): " << std::flush;
    std::cin >> xmlfile;

    return load_dictionary(dict, xmlfile);
}


bool svm_demo_save_dict(Mat dict) {
    string xmlfile;

    std::cout << "Dictionary File (*.xml): " << std::flush;
    std::cin >> xmlfile;

    return save_dictionary(dict, xmlfile);
}


bool svm_demo_prepare_dataset(BOWImgDescriptorExtractor& bowEx,
        Ptr<FeatureDetector> detector, Mat& samples, Mat& labels,int N_classes) {
    string file_list="";
    std::cout << "Please enter File List of training data "<< std::flush;
    std::cin >> file_list;

    std::vector<string> Class_list = read_file_list(file_list);

    string filelist[N_classes];
    for(int i =0;i<N_classes;i++)
    {
    filelist[i]=Class_list[i];
    std::cout << "File List path : " +filelist[i] +"  "<< std::flush;
     _CLASS_LABELS[i]=i;
    std::cout << "Label : " << _CLASS_LABELS[i];

    }


    return prepare_dataset(bowEx, detector,filelist , samples, labels,N_classes);
}


bool svm_demo_save_dataset(Mat samples, Mat labels) {
    string xmlfile;

    std::cout << "Dataset File (*.xml): " << std::flush;
    std::cin >> xmlfile;

    return save_dataset(xmlfile, samples, labels);
}


bool svm_demo_load_dataset(Mat& samples, Mat& labels) {
    string xmlfile;

    std::cout << "Dataset File (*.xml): " << std::flush;
    std::cin >> xmlfile;

    return load_dataset(xmlfile, samples, labels);
}


void svm_demo_test(CvSVM& classifier,
    BOWImgDescriptorExtractor bowEx, Ptr<FeatureDetector> detector) {
    string filelist;

    std::cout << "File List: " << std::flush;
    std::cin >> filelist;

    return testSVM(classifier, bowEx, detector, filelist);
}



void Rtree_demo_test(CvRTrees& classifier,BOWImgDescriptorExtractor bowEx, Ptr<FeatureDetector> detector)
{   string filelist;
    std::cout << "File List: " << std::flush;
    std::cin >> filelist;
 return  testRtree(classifier, bowEx, detector, filelist);

}



