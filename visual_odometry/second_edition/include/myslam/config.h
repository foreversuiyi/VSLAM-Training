#ifndef CONFIG_H
#define CONFIG_H

#include "myslam/common_include.h"

namespace myslam
{
  class Config{
  private:
	static std::shared_ptr<Config> config_;
	cv::FileStorage file_;
	Config(){} //private constructor

  public:
	~Config();  //close the file when deconstructing
	static void setParameterFile(const std::string& filename); //set a new config file

	template<typename T>
	static T get(const std::string& key){return T(Config::config_ -> file_[key]);}  //access the parameter values
  };
}
#endif