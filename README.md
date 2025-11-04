# State_estimation

## Including features
- SLAM/Localization
    - liorf/liorf_localization
- IMU-LiDAR calibration
    - lidar_imu_calib
- Hesai driver
    - HesaiLidar_General_ROS
- YOLO (For traffic sign)
    - utils/yolo_ros1.py
- Editing map
    - Aligning plane to the ground
        - utils/align_plane.py
    - Saving dense map
        - utils/save_dense_points.py

## Log
- 0819
    - Localization module
        - Pre-calculating plane coefficients of prior map
        - Processing time
            - 2 core
                - LiDAR handler loop: 85.3588 ms -> 41.0375 ms
                - odom topic hz: 11.392 -> 16.681
            - 1 core
                - LiDAR handler loop: 364.245 ms -> 187.703 ms
                - odom topic hz: 3.182 -> 5.922
- 0823
    - Add safe mode
        - Filter outliers from imu preintegration results
            - reset bias
            - block outlier initial pose given to the mapOptimization node