cmake_minimum_required( VERSION 3.14 )
set( CMAKE_TOOLCHAIN_FILE "C:/devsoft/vcpkg/scripts/buildsystems/vcpkg.cmake" )

# Create Project
set( PROJECT_NAME "aart" )
project( ${PROJECT_NAME} CUDA CXX )

set(SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

add_executable( ${PROJECT_NAME} ${SRC} )

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

# Set OpenCVConfig.cmake Search Directory
set( OpenCV_DIR "C:/devsoft/vcpkg/installed/x64-windows/share/opencv4" )
# set( OpenCV_DIR )
# if( NOT CMAKE_CL_64 )
#   set( OpenCV_DIR "C:/vcpkg/installed/x86-windows/share/opencv" )
# else()
#   set( OpenCV_DIR "C:/devsoft/vcpkg/installed/x64-windows/share/opencv4" )
# endif()

# Find Package
find_package( OpenCV REQUIRED )

if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Library Directories
  link_directories( ${OpenCV_LIB_DIR} )

  # Additional Dependencies
  target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
endif()