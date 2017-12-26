add_library(myslam SHARED
  camera.cpp
  frame.cpp
  mappoint.cpp
  map.cpp
  config.cpp
  g2o_types.cpp
  visual_odometry.cpp
)

target_link_libraries(myslam ${THIRD_PARTY_LIBS})