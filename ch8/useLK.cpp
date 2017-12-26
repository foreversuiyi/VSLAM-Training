#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <boost/concept_check.hpp>
using namespace cv;

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << "usage: useLK path_to_dataset" << endl;
		return 1;
	}
	string path_to_dataset = argv[1];
	string file = path_to_dataset + "/associate.txt";
	ifstream fin(file);
	if(!fin)
	{
		cerr << "Cannot find associate.txt!" << endl;
		return 1;
	}
	int count = 1;
	string rgb_file, depth_file, time_rgb, time_depth;
	list<Point2f> keypoints;
	Mat color, depth, last_color;
	
	for(int index = 0; index < 500; index++)
	{
		fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
		color = imread(path_to_dataset + "/" + rgb_file);
		depth = imread(path_to_dataset + "/" + depth_file, -1);
		if(index == 0)
		{
			vector<KeyPoint> kps;
			Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
			detector -> detect(color, kps);
			for(auto kp : kps)
				keypoints.push_back(kp.pt);
			last_color = color;
			continue;
		}
		
		if(color.data == nullptr || depth.data == nullptr)
			continue;
		vector<Point2f> next_keypoints;
		vector<Point2f> prev_keypoints;
		for(auto kp:keypoints)
			prev_keypoints.push_back(kp);
		vector<unsigned char> status;
		vector<float> error;
		
		chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
		calcOpticalFlowPyrLK(last_color, color, prev_keypoints, next_keypoints, status, error);
		chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
		chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
		cout << "LK Flow use time: " << time_used.count() << "seconds." << endl;
		
		int i = 0;
		for(auto iter = keypoints.begin(); iter != keypoints.end(); i++)
		{
			if(status[i] == 0)
			{
				iter = keypoints.erase(iter);
				continue;
			}
			*iter = next_keypoints[i];
			iter++;
		}
		
		cout << "tracked keypoints: " << keypoints.size() << endl;
		if(keypoints.size() == 0)
		{
			cout << "all keypoints are lost." << endl;
			break;
		}
		
		Mat img_show = color.clone();
		for(auto kp : keypoints)
			circle(img_show, kp, 5, Scalar(0, 240, 0), 1);
		imshow("corners", img_show);
		waitKey(15);
		last_color = color;
		count ++;
	}
	fin.close();
	cout << endl << "Total pictures: " << count << endl;
	waitKey(0);
	return 0;
}