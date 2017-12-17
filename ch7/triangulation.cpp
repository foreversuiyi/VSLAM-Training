#include <iostream>
#include <boost/concept_check.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(const Mat& img_1, const Mat& img_2, vector<KeyPoint>& keypoints_1,
	vector<KeyPoint>& keypoints_2, vector<DMatch>& matches);

void pose_estimation_2d2d(const vector<KeyPoint>& keypoints_1, const vector<KeyPoint>& keypoints_2,
	const vector<DMatch>& matches, Mat& R, Mat& t);

void triangulation(const vector<KeyPoint>& keypoints_1, const vector<KeyPoint>& keypoints_2, 
	const vector<DMatch>& matches, const Mat& R, const Mat& t, vector<Point3d>& points);

Point2f pixel2cam(const Point2d& p, const Mat& K);

int main(int argc, char** argv)
{
	if (argc != 3)
	{
		cout << "Usage: triangulation img_1 img_2 " << endl;
		return 1;
	}
	Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
	
	vector<KeyPoint> keypoints_1, keypoints_2;
	vector<DMatch> matches;
	find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
	cout << "Totally found " << matches.size() << " matches." << endl;
	
	Mat R, t;
	pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);
	
	vector<Point3d> points;
	triangulation(keypoints_1, keypoints_2, matches, R, t, points);
	
	Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
	for (int i = 0; i < matches.size(); i++)
	{
		Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
		Point2d pt1_cam_3d(points[i].x/points[i].z, points[i].y/points[i].z);
		cout << "Points in the first camera frame : " << pt1_cam << endl;
		cout << "Point projected from 3D " << pt1_cam_3d << ", d = " << points[i].z << endl;
		
		Point2f pt2_cam = pixel2cam(keypoints_2[matches[i].trainIdx].pt, K);
		Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
		pt2_trans /= pt2_trans.at<double>(2, 0);
		cout << "point in the second camera frame : " << pt2_cam << endl;
		cout << "point reprojected from second frame : " << pt2_trans.t() << endl;
		cout << endl;
	}
	
	waitKey(0);
	return 0;
}

void find_feature_matches(const Mat& img_1, const Mat& img_2, vector<KeyPoint>& keypoints_1,
	vector<KeyPoint>& keypoints_2, vector<DMatch>& matches)
{
	Mat descriptors_1, descriptors_2;
	Ptr<FeatureDetector> detector = ORB::create();
	//Ptr<FeatureDetector> detector = FeatureDetector::create("ORB");
	Ptr<DescriptorExtractor> descriptor = ORB::create();
	//Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create("ORB");
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	
	detector -> detect(img_1, keypoints_1);
	detector -> detect(img_2, keypoints_2);
	
	descriptor -> compute(img_1, keypoints_1, descriptors_1);
	descriptor -> compute(img_2, keypoints_2, descriptors_2);
	
	vector<DMatch> dmatch;
	matcher -> match(descriptors_1, descriptors_2, dmatch);
	
	double min_dist = 1000, max_dist = 0;
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = dmatch[i].distance;
		if(dist > max_dist) max_dist = dist;
		if(dist < min_dist) min_dist = dist;
	}
	
	printf("-- Max dist = %f \n", max_dist);
	printf("-- Min dist = %f \n", min_dist);
	
	for(int i = 0; i < descriptors_1.rows; i++)
	{
		if(dmatch[i].distance <= max(2 * min_dist, 30.0))
			matches.push_back(dmatch[i]);
	}
	
	Mat outimg1, outimg2;
	drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("ORB feature points img_1", outimg1);
	
	drawKeypoints(img_2, keypoints_2, outimg2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("ORB feature points img_2", outimg2);
	
	Mat img_match;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
	
	imshow("All match points", img_match);
}

void pose_estimation_2d2d(const vector<KeyPoint>& keypoints_1, const vector<KeyPoint>& keypoints_2,
	const vector<DMatch>& matches, Mat& R, Mat& t)
{
	Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
	vector<Point2f> points1, points2;
	
	for (int i = 0; i < matches.size(); i++)
	{
		points1.push_back(keypoints_1[matches[i].queryIdx].pt);
		points2.push_back(keypoints_2[matches[i].trainIdx].pt);
	}
	
	Mat fundamental_matrix;
	fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
	cout << "Fundamental_matrix is " << endl << fundamental_matrix << endl;
	
	Point2d principal_point(325.1, 249.7);
	int focal_lengh = 521;
	Mat essential_matrix;
	essential_matrix = findEssentialMat(points1, points2, focal_lengh, principal_point);
	cout << "Essential_matrix is " << endl << essential_matrix << endl;
	
	Mat homography_matrix;
	homography_matrix = findHomography(points1, points2, RANSAC, 3);
	cout << "Homography_matrix is " << endl << homography_matrix << endl;
	
	recoverPose (essential_matrix, points1, points2, R, t, focal_lengh, principal_point);
	cout << "R is " << endl << R << endl;
	cout << "t is " << endl << t << endl;
}

void triangulation(const vector<KeyPoint>& keypoints_1, const vector<KeyPoint>& keypoints_2,
	const vector<DMatch>& matches, const Mat& R, const Mat& t, vector<Point3d>& points)
{
	Mat T1 = (Mat_<float>(3, 4) << 
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0);
	Mat T2 = (Mat_<float>(3, 4) <<
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));
	
	Mat K = (Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
	vector<Point2f> pts_1, pts_2;
	for (DMatch m:matches)
	{
		pts_1.push_back(pixel2cam(keypoints_1[m.queryIdx].pt, K));
		pts_2.push_back(pixel2cam(keypoints_2[m.trainIdx].pt, K));
	}
	
	Mat pts_4d;
	triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);
	for(int i = 0; i < pts_4d.cols; i++)
	{
		Mat x = pts_4d.col(i);
		x /= x.at<float>(3, 0);
		Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
		points.push_back(p);
	}
}

Point2f pixel2cam(const Point2d& p, const Mat& K)
{
	return Point2f(
		(p.x - K.at<double>(0, 2))/K.at<double>(0, 0),
		(p.y - K.at<double>(1, 2))/K.at<double>(1, 1)
	);
}
