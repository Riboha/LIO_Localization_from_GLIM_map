import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import pypcd
import math
import matplotlib.pyplot as plt

def load_kitti_poses(txt_path):
    """KITTI 포맷(행 단위 12개 float: m00 m01 m02 tx ... m22 tz)을 Nx(3x4)로 로드"""
    poses = []
    with open(txt_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            vals = list(map(float, s.split()))
            if len(vals) != 12:
                print(f"[WARN] skip line (not 12 floats): {s[:60]}...")
                continue
            M = np.eye(4)
            M[:3, :3] = np.array(vals, dtype=float).reshape(3, 4)[:, :3]
            M[:3, 3]  = np.array(vals, dtype=float).reshape(3, 4)[:, 3]
            poses.append(M)
    return poses  # list of 4x4

def mat4_to_kitti_row(M):
    """4x4 → KITTI 한 줄(12 floats, row-major 3x4) 문자열"""
    Rm = M[:3, :3]
    t  = M[:3, 3]
    row = np.hstack([Rm[0], t[0], Rm[1], t[1], Rm[2], t[2]]).astype(float)
    return " ".join(f"{v:.9f}" for v in row)

def write_pcd_xyzrpy_ascii(filename, xyzrpy_list):
    """
    PCD ASCII로 저장: fields = x y z roll pitch yaw (float32)
    xyzrpy_list: iterable of (x,y,z,roll,pitch,yaw)
    """
    xyzrpy = np.asarray(xyzrpy_list, dtype=np.float32)
    n = xyzrpy.shape[0]
    header = "\n".join([
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z roll pitch yaw",
        "SIZE 4 4 4 4 4 4",
        "TYPE F F F F F F",
        "COUNT 1 1 1 1 1 1",
        f"WIDTH {n}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {n}",
        "DATA ascii",
    ])
    with open(filename, "w") as f:
        f.write(header + "\n")
        for v in xyzrpy:
            f.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*v))

def extract_ground_plane(pcd, distance_threshold, ransac_n, num_iterations):
    plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
    return plane_model

def align_plane_to_xy(plane_model):
    # Normal vector of the plane
    normal = np.array(plane_model[:3])
    normal /= np.linalg.norm(normal)

    # Compute rotation matrix to align with Z-axis
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, z_axis)
    angle = np.arccos(np.dot(normal, z_axis))

    if np.linalg.norm(axis) < 1e-6:
        return np.eye(4)

    axis /= np.linalg.norm(axis)
    rot = R.from_rotvec(axis * angle).as_matrix()

    T = np.eye(4)
    T[:3, :3] = rot
    return T

def translate_plane_to_z0(plane_model, T_rot):
    point_on_plane = -plane_model[3] * np.array(plane_model[:3])  # ax + by + cz + d = 0 → point = -d * n
    point_on_plane = np.append(point_on_plane, 1.0)
    transformed_point = T_rot @ point_on_plane
    z_shift = transformed_point[2]

    T = np.eye(4)
    T[2, 3] = -z_shift
    return T

def find_densest_z_slice(pcd, bin_size=0.05):
    """
    Finds the densest z-slice in a point cloud and visualizes the histogram.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        bin_size (float): The size of the bins for the histogram.

    Returns:
        float: The center Z-coordinate of the densest slice.
    """
    z_vals = np.asarray(pcd.points)[:, 2]

    # Create the histogram
    hist, bin_edges = np.histogram(z_vals, bins=np.arange(z_vals.min(), z_vals.max(), bin_size))
    
    # Find the bin with the most points
    max_bin_index = np.argmax(hist)
    
    # Calculate the center of the densest slice
    z_center = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2.0
    
    print(f"Densest Z slice found at: {z_center:.4f}")
    
    # --- Visualization Code ---
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], hist, width=bin_size, align='edge', color='skyblue', edgecolor='black')
    
    # Highlight the densest bin
    plt.bar(bin_edges[max_bin_index], hist[max_bin_index], width=bin_size, color='red', edgecolor='black', label=f'Densest Slice (z={z_center:.4f})')
    
    plt.title('Z-axis Point Distribution')
    plt.xlabel('Z-coordinate')
    plt.ylabel('Number of Points')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return z_center

def transform_xyz_rpy(xyz_rpy, T_total):
    xyz = xyz_rpy[:3]
    rpy = xyz_rpy[3:]

    # Convert RPY to rotation matrix
    rot = R.from_euler('xyz', rpy).as_matrix()

    # Compose full transform
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = xyz

    # Apply total transform
    T_new = T_total @ T

    # Decompose back to xyz, rpy
    new_xyz = T_new[:3, 3]
    new_rpy = R.from_matrix(T_new[:3, :3]).as_euler('xyz')

    return np.concatenate([new_xyz, new_rpy])

# ---------------------- Main ---------------------------
def main():
    map_dir = "/home/riboha/posco/demo_test"
    input_pcd_sparse = os.path.join(map_dir, "GlobalMap.pcd")
    input_pcd_surf = os.path.join(map_dir, "SurfMap.pcd")
    input_transformations = os.path.join(map_dir, "keyframe_poses.txt")
    input_pcd_dense = "/home/unitree/pointcloud_to_2dmap/posco_challenge/GlobalMap_dense.pcd"

    output_maps_dir = os.path.join(map_dir, "aligned")
    output_pcd_sparse = os.path.join(output_maps_dir, "GlobalMap.pcd")
    output_pcd_surf = os.path.join(output_maps_dir, "SurfMap.pcd")
    output_transformations = os.path.join(output_maps_dir, "transformations.pcd")
    output_pcd_dense = os.path.join(output_maps_dir, "GlobalMap_dense.pcd")

    print("🔹 Loading point cloud...")
    full_pcd = o3d.io.read_point_cloud(input_pcd_sparse)
    print(f"✅ Loaded point cloud with {len(full_pcd.points)} points.")

    raw_points = np.asarray(full_pcd.points)
    filtered_points = raw_points[raw_points[:, 2] <= 3.0]
    full_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    print(f"🔹 Filtered: {len(filtered_points)} points remaining (z <= 3.0)")

    voxel_size = 0.1
    pcd_down = full_pcd.voxel_down_sample(voxel_size)
    print(f"🔹 Downsampled to {len(pcd_down.points)} points for plane extraction")

    plane_model = extract_ground_plane(pcd_down, distance_threshold=0.02, ransac_n=1000, num_iterations=10000)
    T_rot = align_plane_to_xy(plane_model)
    T_trans = translate_plane_to_z0(plane_model, T_rot)

    # T_total = T_final @ T_trans @ T_rot  # Apply same transform to other inputs
    T_total = T_trans @ T_rot
    full_pcd.transform(T_total)

    z_center = find_densest_z_slice(full_pcd)
    T_final = np.eye(4)
    margin = 0.0
    T_final[2, 3] = -z_center + margin
    # full_pcd.transform(T_final)
    
    T_total = T_final @ T_trans @ T_rot

    print(f"✅ Final correction: shifted point cloud by {-z_center:.4f} in Z to align bottom with Z=0")

    # ---------------- Transform & Save others ----------------
    # SurfMap
    pcd_surf = o3d.io.read_point_cloud(input_pcd_surf)
    pcd_surf.transform(T_total)
    o3d.io.write_point_cloud(output_pcd_surf, pcd_surf)
    print(f"✅ Saved aligned surf map to {output_pcd_surf}")

    # DenseMap
    pcd_dense = o3d.io.read_point_cloud(input_pcd_dense)
    pcd_dense.transform(T_total)
    o3d.io.write_point_cloud(output_pcd_dense, pcd_dense)
    print(f"✅ Saved aligned dense map to {output_pcd_dense}")

    # Transformations file
    poses_4x4 = load_kitti_poses(input_transformations)

    aligned_poses_4x4 = []
    xyzrpy_list = []
    for M in poses_4x4:
        M_aligned = T_total @ M
        aligned_poses_4x4.append(M_aligned)

        # 6D (x,y,z,roll,pitch,yaw) 추출 (RPY는 'xyz' 순, 라디안)
        x, y, z = M_aligned[:3, 3]
        rpy = R.from_matrix(M_aligned[:3, :3]).as_euler('xyz', degrees=False)
        xyzrpy_list.append([x, y, z, rpy[0], rpy[1], rpy[2]])

    # 2-1) KITTI 형식으로도 저장
    os.makedirs(output_maps_dir, exist_ok=True)
    kitti_out = os.path.join(output_maps_dir, "keyframe_poses.txt")
    with open(kitti_out, "w") as f:
        for M in aligned_poses_4x4:
            f.write(mat4_to_kitti_row(M) + "\n")
    print(f"✅ Saved aligned trajectory (KITTI) to {kitti_out}")

    # 2-2) PCD(x,y,z,roll,pitch,yaw)로 저장 (ROS에서 읽기 쉬움)
    write_pcd_xyzrpy_ascii(output_transformations, xyzrpy_list)
    print(f"✅ Saved aligned trajectory (PCD xyzrpy) to {output_transformations}")

    # Save sparse-aligned
    o3d.io.write_point_cloud(output_pcd_sparse, full_pcd)
    print(f"✅ Saved aligned point cloud to:\n  {output_pcd_sparse}")

    # Final Z Range
    z_points = np.asarray(full_pcd.points)[:, 2]
    print(f"📐 Final Z range: {z_points.min():.4f} to {z_points.max():.4f}")

    # Visualize
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([full_pcd, axis])

if __name__ == "__main__":
    main()
