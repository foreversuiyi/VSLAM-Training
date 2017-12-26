add_library(myslam SHARED
  camera.cpp
  frame.cpp
  mappoint.cpp
  map.cpp
  config.cpp
)

target_link_libraries(myslam ${THIRD_PARTY_LIBS})