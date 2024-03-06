import os
import open3d as o3d
import numpy as np

def load_point_clouds_from_folder(folder_path):
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
    return transformation_icp, information_icp, source_down, target_down

def print_transformation_details(point_clouds, transformation_icp, i, source_normals, target_normals):
    source_filename, source = point_clouds[i-1]
    target_filename, target = point_clouds[i]

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

def merge_point_clouds(point_clouds, threshold=0.1):
    # In ra thông tin về quá trình ghép các mặt
    for i in range(1, len(point_clouds)):
        source_filename, source = point_clouds[i-1]
        target_filename, target = point_clouds[i]

        # Extract thông tin từ tên file
        source_name, source_part, source_face = extract_info_from_filename(source_filename)
        target_name, target_part, target_face = extract_info_from_filename(target_filename)

        # Kiểm tra xem hai mảnh có giống nhau không
        if source_name != target_name and source_part != target_part:
            # Tính toán sai số sau khi ghép nối
            transformation_icp, information_icp, source_down, target_down = register_point_clouds(source, target, voxel_size=0.05)
            fitness = o3d.pipelines.registration.evaluate_registration(
                source_down, target_down, threshold, transformation_icp
            ).fitness

            # In ra thông tin
            print(f"({source_name} - Mặt {source_face}) được ghép với ({target_name} - Mặt {target_face}) với sai số: {fitness}")

            # Kiểm tra xem sai số có dưới ngưỡng không
            if fitness < threshold:
                # Áp dụng biến đổi để ghép các mặt
                source.transform(transformation_icp)

                # In ra chi tiết về biến đổi và pháp tuyến
                print_transformation_details(point_clouds, transformation_icp, i, source_down, target_down)

    # Tạo một đám mây điểm lớn từ các mặt đã được ghép nối
    merged_point_cloud = point_clouds[0][1]
    for i in range(1, len(point_clouds)):
        merged_point_cloud += point_clouds[i][1]

    return merged_point_cloud


def main():
    folder_path = "C:/group project/icp/cake"
    point_clouds = load_point_clouds_from_folder(folder_path)
    merged_point_cloud = merge_point_clouds(point_clouds)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 2
    vis.get_view_control().set_lookat([0, 0, 0])
    vis.get_view_control().set_up([0, -1, 0])
    vis.get_view_control().set_zoom(0.8)

    vis.add_geometry(merged_point_cloud)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()

