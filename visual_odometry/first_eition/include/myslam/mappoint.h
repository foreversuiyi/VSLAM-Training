#ifndef MAPPOINT_H
#define MAPPOINT_H

namespace myslam
{
  class Frame;
  class MapPoint{
  public:
	typedef shared_ptr<MapPoint> Ptr;
	unsigned long id_;	//ID
	Vector3d pos_;		//Word position
	Vector3d norm_;		//Normal of viewing direction
	Mat descriptor_;	//Descriptor of matching
	int observed_times_;	//Been observed
	int correct_times_;		//Been matched
	
	MapPoint();
	MapPoint(long id, Vector3d position, Vector3d norm);
	//factory function
	static MapPoint::Ptr createMapPoint();
  };
}

#endif