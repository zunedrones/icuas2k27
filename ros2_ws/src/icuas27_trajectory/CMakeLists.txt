cmake_minimum_required(VERSION 3.8)
project(icuas27_trajectory)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(laser_msgs REQUIRED)
find_package(icuas27_msgs REQUIRED)
find_package(pmm_planner REQUIRED)
find_package(super_planner REQUIRED)

add_executable(trajectory_predictor src/trajectory_predictor_node.cpp)
# Same definitions as the LUS planning manager uses for the super_planner (MINCO) headers.
target_compile_definitions(trajectory_predictor PRIVATE USE_ROS2 FMT_HEADER_ONLY)
ament_target_dependencies(trajectory_predictor rclcpp geometry_msgs laser_msgs icuas27_msgs
  super_planner Eigen3)
target_link_libraries(trajectory_predictor pmm_planner::pmm_planner)

install(TARGETS trajectory_predictor DESTINATION lib/${PROJECT_NAME})

ament_package()
