cmake_minimum_required(VERSION 3.5)
project(grav_align)
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
install(
    DIRECTORY
    launch
    DESTINATION share/${PROJECT_NAME}
)
install(
    PROGRAMS
    scripts/imu_gravity_aligner.py
    DESTINATION lib/${PROJECT_NAME}
)
ament_package()