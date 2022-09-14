cmake_minimum_required(VERSION 3.5)
project(cv_targeting)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
include_directories( ${OpenCV_INCLUDE_DIRS} )
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(teleop_interfaces REQUIRED)
find_package( OpenCV REQUIRED PATHS /opencv/build/OpenCVConfig.cmake ) 

add_executable(target src/target.cpp)
target_link_libraries( target ${OpenCV_LIBS} )
ament_target_dependencies(target rclcpp std_msgs teleop_interfaces OpenCV)

install(TARGETS
	target
	DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


ament_package()
