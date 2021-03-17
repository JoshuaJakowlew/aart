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
#set( OpenCV_DIR "C:/devsoft/vcpkg/installed/x64-windows/share/opencv4" )
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

find_package(freetype CONFIG REQUIRED)
if( freetype_FOUND )
  include_directories( ${freetype_INCLUDE_DIRS} )
  link_directories( ${freetype_LIB_DIR} )
  target_link_libraries( ${PROJECT_NAME} freetype)
endif()

# find_package(Freetype CONFIG) # should find freetype-config.cmake if available
# find_package(Freetype REQUIRED) # Will not be executed if Freetype_FOUND ist already set
# # if you do not want two find_package calls consider using CMAKE_FIND_PACKAGE_PREFER_CONFIG

# if(TARGET freetype AND NOT TARGET Freetype::Freetype)
#      add_library(Freetype::Freetype ALIAS freetype) # target freetype is defined by freetype-targets.cmake
#      # might need to add freetype to global scope if cmake errors here
#      # alternativly if the above does not work for you you can use
#      # add_library(Freetype::Freetype INTERFACE IMPORTED)
#      # target_link_libraries(Freetype::Freetype INTERFACE freetype)
#  endif()
#  if(NOT TARGET Freetype::Freetype)
#      # insert error here
#      # or create the target correctly (see cmakes newer FindFreetype.cmake)
#  endif()
#  target_link_libraries(mylib PRIVATE Freetype::Freetype)