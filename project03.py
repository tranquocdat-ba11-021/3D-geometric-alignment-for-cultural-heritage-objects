import os
import open3d as o3d
import numpy as np

def load_ply_files(folder_path):
    point_clouds = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".ply"):
            file_path = os.path.join(folder_path, filename)
            pcd = o3d.io.read_point_cloud(file_path)
            point_clouds.append((filename, pcd))
    return point_clouds

def extract_info_from_filename(filename):
    parts = filename.split("-")

    if len(parts) < 3:
        raise ValueError("Invalid filename format")

    object_name = parts[0]
    
    part_info = parts[1].split("[")
    if len(part_info) < 2:
        part_number = 0
    else:
        part_number_str = part_info[1].split("]")[0]
        part_number = int(part_number_str) if part_number_str.isdigit() else 0

    face_number_str = parts[2].split("]")[0]
    face_number = int(face_number_str) if face_number_str.isdigit() else 0
    
    return object_name, part_number, face_number

def preprocess_point_cloud_with_normals(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    return pcd_down

def register_point_clouds(source, target, voxel_size):
    source_down = preprocess_point_cloud_with_normals(source, voxel_size)
    target_down = preprocess_point_cloud_with_normals(target, voxel_size)

    result_icp = o3d.pipelines.registration.registration_icp(
        source_down, target_down, 0.05, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )

    transformation_icp = result_icp.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source_down, target_down, voxel_size, result_icp.transformation
    )
    return transformation_icp, information_icp, source_down, target_down, result_icp.fitness

def print_transformation_details(source_filename, target_filename, transformation_icp, source_normals, target_normals):
    source_name, source_part, source_face = extract_info_from_filename(source_filename)
    target_name, target_part, target_face = extract_info_from_filename(target_filename)
    
    print(f"=============== Chi Tiết Biến Đổi từ ({source_name} - Mặt {source_face}) đến ({target_name} - Mặt {target_face}) ========================")

    translation = transformation_icp[:3, 3]
    rotation_matrix = transformation_icp[:3, :3]

    print("\nMa trận Biến Đổi:")
    print(transformation_icp)

    print("\nDịch Chuyển:")
    print(translation)

    print("\nMa Trận Quay:")
    print(rotation_matrix)

    euler_angles = np.degrees(np.array([
        np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]),
        np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2)),
        np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    ]))

    print("\nGóc Euler (đơn vị độ):")
    print(euler_angles)

    print(f"\nSố lượng pháp tuyến trước khi đăng ký (source_normals): {len(source_normals.normals)}")
    print(f"Số lượng pháp tuyến sau khi đăng ký (target_normals): {len(target_normals.normals)}")

    print("====================== Kết Thúc Chi Tiết Biến Đổi ======================\n")

def pairwise_icp(point_clouds, threshold=0.1):
    num_point_clouds = len(point_clouds)

    for i in range(num_point_clouds):
        source_filename, source = point_clouds[i]

        for j in range(i+1, num_point_clouds):
            target_filename, target = point_clouds[j]

            transformation_icp, information_icp, source_down, target_down, fitness = register_point_clouds(source, target, voxel_size=0.05)

            print(f"({source_filename}) so với ({target_filename}) - Thành công: {fitness < threshold}, Threshold: {fitness}")

            if fitness < threshold:
                print_transformation_details(source_filename, target_filename, transformation_icp, source_down, target_down)

    # Merge all point clouds
    merged_point_cloud = o3d.geometry.PointCloud()
    for _, pcd in point_clouds:
        merged_point_cloud += pcd

    # Visualize the merged point cloud
    o3d.visualization.draw_geometries([merged_point_cloud])

def main():
    folder_path = "C:/group project/icptest/cake"
    subfolders = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    all_point_clouds = []
    for subfolder in subfolders:
        point_clouds = load_ply_files(subfolder)
        all_point_clouds.extend(point_clouds)

    pairwise_icp(all_point_clouds, threshold=0.1)

if __name__ == "__main__":
    main()
