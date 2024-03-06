
import os
import open3d as o3d

def visualize_ply_file(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    o3d.visualization.draw_geometries([pcd])

def load_and_visualize_ply_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".ply"):
            file_path = os.path.join(folder_path, filename)
            visualize_ply_file(file_path)

def main():
    # Đường dẫn đến thư mục chứa các tệp tin .ply
    folder_path = "C:/group project/icptest/caketest/part01"

    # Tải và hiển thị từng điểm đám mây 3D
    load_and_visualize_ply_files(folder_path)

if __name__ == "__main__":
    main()
