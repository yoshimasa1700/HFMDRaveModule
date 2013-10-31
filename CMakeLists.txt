cmake_minimum_required (VERSION 2.6.0)
project (HFMDRaveModule)

find_package(OpenRAVE REQUIRED)

if( CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX )
  add_definitions("-fno-strict-aliasing -Wall")
endif( CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX )

find_package(Boost ${OpenRAVE_Boost_VERSION} EXACT COMPONENTS iostreams python thread)
find_package(OpenCV REQUIRED)
find_package(OpenMP)

include_directories(${OpenRAVE_INCLUDE_DIRS} 
					     ${CMAKE_CURRENT_SOURCE_DIR} 
					     ${Boost_INCLUDE_DIRS} 
					     ${OpenCV_INCLUDE_DIRS} 
					     /usr/local/include 
					     /opt/softkinetic/DepthSenseSDK/include)

link_directories(${OpenRAVE_LIBRARY_DIRS} 
					  ${Boost_LIBRARY_DIRS} 
					  ${OpenCV_LIB_DIR} 
					  "/usr/local/lib"  
					  "/opt/softkinetic/DepthSenseSDK/lib")
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_library(HFMDRaveModule SHARED 
			   ./src/HFMDRaveModule.cpp
			   ./src/CCalibDS325.cpp)
set_target_properties(HFMDRaveModule PROPERTIES COMPILE_FLAGS "${OpenRAVE_CXX_FLAGS}")
set_target_properties(HFMDRaveModule PROPERTIES LINK_FLAGS "${OpenRAVE_LINK_FLAGS}")
target_link_libraries(HFMDRaveModule 
				     HFMD_core
				     ${OpenRAVE_LIBRARIES}
				     DepthSense
				     DepthSensePlugins
				     turbojpeg
				     )
install(TARGETS HFMDRaveModule DESTINATION ../share/openrave-0.8/plugins)
