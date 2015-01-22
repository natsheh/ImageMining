#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
/*
 * File:   BOWToyLib.h
 * Author: Andrea Moio
 *
 * Created on 29 Novembre 2013, 12.56
 *
 * Simple 2-class SVM classifier based
 * on Bag of Visual Words Model
 *
 *
 */


#include <opencv2/ml/ml.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <utility>
#include <cmath>

using namespace cv;

/*********************
    Global variables
*********************/
//class labels
extern std::string _CLASS_LABELS[1000];
const std::string WIN_NAME = "ImageClassification";



/*********************
    Functions
*********************/
//read an ascii file containing a list of absolute pathnames
std::vector<string> read_file_list(const string file_list);

//create a dictionary in BOW Model
Mat create_dictionary(const string file_list);

//save a dictionary in xml format
bool save_dictionary(Mat dict, const string xmlfile);

//load a dictionary from an xml file
bool load_dictionary(Mat& dict, const string xmlfile);

//SVM training set creation
bool prepare_dataset(BOWImgDescriptorExtractor& bowEx, Ptr<FeatureDetector> detector,
					 string list1[], Mat& samples, Mat& labels);

//load a pre-prepared dataset from an xmlfile
bool load_dataset(const string filename, Mat& samples, Mat& labels);

//save a dataset in xml format
bool save_dataset(const string filename, Mat samples, Mat labels);

//SVM training function
bool trainSVM(CvSVM& classifier, Mat samples, Mat labels);

//SVM test function
void testSVM(CvSVM& classifier, BOWImgDescriptorExtractor bowEx,
					Ptr<FeatureDetector> detector, const string file_list);
