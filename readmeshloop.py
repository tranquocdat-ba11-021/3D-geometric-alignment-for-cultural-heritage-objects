import os
import open3d as o3d
import pyvista as pv
import fast_simplification

# Đường dẫn thư mục chứa tệp PCD
pcd_folder_path = "C:/group project/pcd/Sculpture pcd"

# Đường dẫn thư mục chứa tệp OBJ
obj_folder_path = "C:/group project/obj/sculpture obj"

# Lấy danh sách tất cả các tệp PCD trong thư mục
pcd_files = [f for f in os.listdir(pcd_folder_path) if f.endswith(".pcd")]

# Lặp qua từng tệp PCD và thực hiện chuyển đổi sang tệp OBJ
for pcd_file in pcd_files:
    # Tạo đường dẫn đầy đủ đến tệp PCD
    pcd_file_path = os.path.join(pcd_folder_path, pcd_file)

    # Đọc điểm đám mây từ tệp PCD với định dạng cụ thể
    point_cloud = o3d.io.read_point_cloud(pcd_file_path, format='xyzn')

    # Xây dựng mesh từ điểm đám mây
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud)

    # Tạo đường dẫn đầy đủ đến tệp OBJ
    obj_file_path = os.path.join(obj_folder_path, os.path.splitext(pcd_file)[0] + ".obj")

    # Xuất mesh ra tệp OBJ
    o3d.io.write_triangle_mesh(obj_file_path, mesh)

    print(f"Mesh của {pcd_file} đã được xuất thành công vào tệp {obj_file_path}")

    # Đọc mesh từ tệp OBJ
    mesh_pv = pv.read(obj_file_path)

    # Đơn giản hóa mesh
    mesh_pv = fast_simplification.simplify_mesh(mesh=mesh_pv, target_reduction=0.9)

    # Hiển thị mesh
    plotter = pv.Plotter()
    plotter.add_mesh(mesh_pv, color="lightgrey", show_edges=True)
    plotter.show()
