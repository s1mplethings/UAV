import open3d as o3d

pcd = o3d.io.read_point_cloud(r"D:\.py_projects\UAV\UAV\outputs\delivery_area\dense\fused.ply")
o3d.visualization.draw_geometries([pcd])