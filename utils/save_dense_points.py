#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

import threading
import numpy as np
import open3d as o3d
import time
import select
import sys
import tty
import termios
import os


class PointCloudMerger(object):
    def __init__(self):
        self.sub = rospy.Subscriber(
            '/liorf/mapping/cloud_registered_raw',
            PointCloud2,
            self.pointcloud_callback,
            queue_size=10
        )

        self.lock = threading.Lock()
        self.merged_pcd = o3d.geometry.PointCloud()
        rospy.loginfo("Subscribed to /liorf/mapping/cloud_registered_raw")

        # Keypress thread
        self.thread = threading.Thread(target=self.keypress_listener, daemon=True)
        self.thread.start()

    def pointcloud_callback(self, msg: PointCloud2):
        # PointCloud2 -> numpy (x,y,z)
        cloud_iter = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        cloud_array = np.fromiter(cloud_iter, dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
        if cloud_array.size == 0:
            return

        pts = np.vstack((cloud_array['x'], cloud_array['y'], cloud_array['z'])).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        with self.lock:
            self.merged_pcd += pcd
        rospy.loginfo(f"PointCloud received and merged: {pts.shape[0]} points")

    def keypress_listener(self):
        # 터미널 raw 모드
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        rospy.loginfo("Press 'q' to save merged pointcloud as GlobalMap_allframe.pcd")

        try:
            while not rospy.is_shutdown():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key == 'q':
                        self.save_pointcloud()
                        break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def save_pointcloud(self):
        # 저장 경로
        filename = '/home/unitree/agile_maps/0812_demo2/GlobalMap_dense.pcd'
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with self.lock:
            if len(self.merged_pcd.points) > 0:
                o3d.io.write_point_cloud(filename, self.merged_pcd)
                rospy.loginfo(f"Saved merged pointcloud as {filename}")
            else:
                rospy.logwarn("Merged pointcloud is empty, nothing to save.")


def main():
    rospy.init_node('pointcloud_merger', anonymous=False)
    merger = PointCloudMerger()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # rospy는 명시적 shutdown 불필요하지만, 깔끔하게 종료 로그만 출력
        rospy.loginfo("Shutting down pointcloud_merger")


if __name__ == '__main__':
    main()
