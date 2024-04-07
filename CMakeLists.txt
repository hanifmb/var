cmake_minimum_required(VERSION 2.8)
project( main )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${CMAKE_SOURCE_DIR}/include/alglib )

file(GLOB ALGLIB
     "include/alglib/*.h"
     "include/alglib/*.cpp"
     "src/*.h"
     "src/*.cpp"
)

add_executable( main ${ALGLIB})
target_link_libraries( main ${OpenCV_LIBS} )