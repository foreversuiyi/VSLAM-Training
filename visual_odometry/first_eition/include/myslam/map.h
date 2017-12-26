#ifndef MAP_H
#define MAP_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/mappoint.h"

namespace myslam
{
  class Map{
  public:
	typedef shared_ptr<Map> Ptr;
	unordered_map<unsigned long, MapPoint::Ptr> map_points_;  //landmarks
	unordered_map<unsigned long, Frame::Ptr> keyframes_;		//key-frames
	
	Map(){}
	
	void insertKeyFrame(Frame::Ptr frame);
	void insertMapPoint(MapPoint::Ptr map_point);
  };
}

#endif