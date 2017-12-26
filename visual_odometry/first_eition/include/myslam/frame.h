#ifndef FRAME_H
#define FRAME_H

#include "myslam/common_include.h"
#include "myslam/camera.h"

namespace myslam
{
class MapPoint;
class Frame{
public:
  typedef std::shared_ptr<Frame> Ptr;
  unsigned long id_;   //The ID of this frame
  double time_stamp_;  //The time when this frame is recorded
  SE3 T_c_w_;			//The pose of the camera for this frame
  Camera::Ptr camera_;  //RGBD Camera model
  Mat color_, depth_;	//Color and Depth image data
  
public: //data members
  Frame();
  Frame(long id, double time_stamp = 0, SE3 T_c_w = SE3(), Camera::Ptr camera = nullptr, Mat color = Mat(), Mat depth = Mat());
  ~Frame();
  
  //factory function
  static Frame::Ptr createFrame();
  //Get the depth data of one keypoint
  double findDepth(const cv::KeyPoint& kp);
  //Get camera center
  Vector3d getCamCenter() const;
  //check if a point is in the frame
  bool isInFrame(const Vector3d& pt_world);
};
}

#endif