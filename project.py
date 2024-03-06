import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

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

def register_point_clouds(source, target, voxel_size, max_iterations=200):
    source_down = preprocess_point_cloud_with_normals(source, voxel_size)
    target_down = preprocess_point_cloud_with_normals(target, voxel_size)

    fitness_values = []  # Thêm một danh sách để lưu trữ giá trị fitness qua các vòng lặp

    for iteration in range(max_iterations):
        result_icp = o3d.pipelines.registration.registration_icp(
            source_down, target_down, 0.05, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1)
        )

        fitness_values.append(result_icp.fitness)  # Lưu giá trị fitness

    transformation_icp = result_icp.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source_down, target_down, voxel_size, transformation_icp
    )
    return transformation_icp, information_icp, source_down, target_down, fitness_values

def print_transformation_details(source_filename, target_filename, transformation_icp, source_normals, target_normals):
    source_name, source_part, source_face = extract_info_from_filename(source_filename)
    target_name, target_part, target_face = extract_info_from_filename(target_filename)
    
    print(f"=============== Details change from ({source_name} - face {source_face}) to ({target_name} - face {target_face}) ========================")

    translation = transformation_icp[:3, 3]
    rotation_matrix = transformation_icp[:3, :3]

    print("\nMa trận Biến Đổi:")
    print(transformation_icp)

    print("\nTransformation Matrix:")
    print(translation)

    print("\nRotation Matrix:")
    print(rotation_matrix)

    euler_angles = np.degrees(np.array([
        np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]),
        np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2)),
        np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    ]))

    print("\nEuler angle (Degree):")
    print(euler_angles)

    print(f"\nNumber of normals before registration (source_normals): {len(source_normals.normals)}")
    print(f"Number of normals after registration (target_normals): {len(target_normals.normals)}")

    print("====================== End of details change ======================\n")

def pairwise_icp(point_clouds):
    num_point_clouds = len(point_clouds)
    
    # List to store previously selected thresholds
    previous_thresholds = []
    
    # List to store fitness values for all pairs
    all_fitness_values = []

    for i in range(num_point_clouds):
        source_filename, source = point_clouds[i]

        for j in range(i+1, num_point_clouds):
            target_filename, target = point_clouds[j]

            # Choose threshold based on the average of previously selected thresholds
            threshold = np.mean(previous_thresholds) if previous_thresholds else 0.1

            transformation_icp, information_icp, source_down, target_down, fitness_values = register_point_clouds(source, target, voxel_size=0.05)

            print(f"({source_filename}) compare with ({target_filename}) - Status: {fitness_values[-1] < threshold}, Fitness: {fitness_values[-1]}")

            if fitness_values[-1] < threshold:
                print_transformation_details(source_filename, target_filename, transformation_icp, source_down, target_down)
                # Add the current threshold to the list of previous thresholds
                previous_thresholds.append(fitness_values[-1])

            all_fitness_values.append(fitness_values)

    # Merge all point clouds
    merged_point_cloud = o3d.geometry.PointCloud()
    for _, pcd in point_clouds:
        merged_point_cloud += pcd

    # Visualize the merged point cloud
    o3d.visualization.draw_geometries([merged_point_cloud])

    # Plotting the fitness values over iterations
    plot_fitness_values(all_fitness_values)

def plot_fitness_values(all_fitness_values):
    num_pairs = len(all_fitness_values)
    num_iterations = len(all_fitness_values[0])

    plt.figure(figsize=(10, 6))
    for pair_index in range(num_pairs):
        plt.plot(range(num_iterations), all_fitness_values[pair_index], label=f"Pair {pair_index + 1}")

    plt.title("Fitness Values Over Iterations")
    # plt.xlabel("Iterations")
    plt.ylabel("Fitness Value")
    plt.legend()
    plt.show()

def main():
    folder_path = "C:/group project/icptest/cake"
    subfolders = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    all_point_clouds = []
    for subfolder in subfolders:
        point_clouds = load_ply_files(subfolder)
        all_point_clouds.extend(point_clouds)

    pairwise_icp(all_point_clouds)

if __name__ == "__main__":
    main()

