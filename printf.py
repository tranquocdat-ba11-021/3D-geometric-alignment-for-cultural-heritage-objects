import open3d as o3d
import numpy as np

# Đường dẫn đến tệp PCD
pcd_file_path = "C:/group project/pcd/cake pcd/cake_part01.pcd"

# Đọc điểm đám mây từ tệp PCD với định dạng 'xyz'
point_cloud = o3d.io.read_point_cloud(pcd_file_path, format='xyz')

# Hiển thị điểm đám mây
o3d.visualization.draw_geometries([point_cloud])
