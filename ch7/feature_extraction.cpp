#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	if (argc != 3)
	{
		cout << "Usage: feature_extraction imag1 imag2" << endl;
		return 1;
	}
	Mat imag_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat imag_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
	
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	Ptr<FeatureDetector> detector = ORB::create();
	Ptr<DescriptorExtractor> descriptor = ORB::create();
	//Ptr<FeatureDetector> detector = FeatureDetector::create(detector_name);
	//Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::creator(descriptor_name);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	
	detector -> detect(imag_1, keypoints_1);
	detector -> detect(imag_2, keypoints_2);
	
	descriptor -> compute(imag_1, keypoints_1, descriptors_1);
	descriptor -> compute(imag_2, keypoints_2, descriptors_2);
	
	Mat outimg1;
	drawKeypoints(imag_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("ORB feature points", outimg1);
	
	vector<DMatch> matches;
	matcher -> match(descriptors_1, descriptors_2, matches);
	double min_dist = 10000, max_dist = 0;
	
	for(int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;
		if(dist < min_dist) min_dist = dist;
		if(dist > max_dist) max_dist = dist;
	}
	
	min_dist = min_element(matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2)
	{return m1.distance < m2.distance;}) -> distance;
	max_dist = max_element(matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2)
	{return m1.distance < m2.distance;}) -> distance;
	printf("--Max dist: %f \n", max_dist);
	printf("--Min dist: %f \n", min_dist);
	
	vector<DMatch> good_matches;
	for(int i = 0; i < descriptors_1.rows; i++)
	{
		if(matches[i].distance <= max(2 * min_dist, 30.0))
			good_matches.push_back(matches[i]);
	}
	
	Mat img_match;
	Mat img_good_match;
	drawMatches(imag_1, keypoints_1, imag_2, keypoints_2, matches, img_match);
	drawMatches(imag_1, keypoints_1, imag_2, keypoints_2, good_matches, img_good_match);
	
	imshow("All match points", img_match);
	imshow("All good_match points after optimization", img_good_match);
	
	waitKey(0);
	
	return 0;
}