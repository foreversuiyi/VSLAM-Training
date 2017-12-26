#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include "myslam/common_include.h"
#include "myslam/map.h"
#include <opencv2/features2d/features2d.hpp>

namespace myslam
{
  class VisualOdometry{
  public:
	typedef shared_ptr<VisualOdometry> Ptr;
	enum VOState{
	  INITIALIZING =-1,
	  OK = 0,
	  LOST
	};
	VOState state_;		//Current VO status
	Map::Ptr map_;		//All frames and mappoints
	
	Frame::Ptr ref_;	//Reference frame
	Frame::Ptr curr_;	//Current frame
	
	cv::Ptr<cv::ORB> orb_;	//ORB detector and computer
	vector<cv::KeyPoint> keypoints_curr_;	//Keypoints in current frame
	//vector<cv::KeyPoint> keypoints_ref_; 	//Keypoints in reference frame
	Mat descriptors_curr_;	//Descriptor in current frame
	//Mat descriptors_ref_;	//Descriptor in reference frame
	//vector<cv::DMatch> feature_matches_;
	
	cv::FlannBasedMatcher matcher_flann_;   //flann matcher
	vector<MapPoint::Ptr> match_3dpts_;		//matched 3d points
	vector<int> match_2dkp_index_;			//matched 2d pixels(index of kp_curr)
	
	//SE3 T_c_r_estimated_;	//Estimated pose of current frame
	SE3 T_c_w_estimated_;	//Estimated pose of current frame
	int num_inliers_;		//Number of inlier features in ICP
	int num_lost_;			//number of lost times
	
	int num_features_;	//number of features
	double scale_;		//scale in image pyramid
	int level_pyramid_;	//number of pyramid levels
	float match_ratio_;	//Ratio of selecting good matches
	int max_num_lost_;	//Max number of continuous lost times
	int min_inliers_;	//minimum inliers
	
	double key_frame_min_rot;	//minimal rotation of two key frames
	double key_frame_min_trans;	//minimal translation of two key frames
	double map_point_erase_ratio_;	//remove map point ratio
	
  public:
	VisualOdometry();
	~VisualOdometry();
	bool addFrame(Frame::Ptr frame);
	
  protected:
	
	void extractKeyPoints();
	void computeDescriptors();
	void featureMatching();
	void poseEstimationPnP();
	
	void optimizeMap();
	//void setRef3DPoints();
	
	void addKeyFrame();
	void addMapPoints();
	bool checkEstimatedPose();
	bool checkKeyFrame();
	
	double getViewAngle(Frame::Ptr frame, MapPoint::Ptr point);
  };
}

#endif