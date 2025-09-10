import open3d as o3d
import os
pcd_path = "/home/ubuntu/DATA1/junhaoge/sg_assets/agent_ply_ori/"

for file in os.listdir(pcd_path):
    if file.endswith(".ply"):
        # 读取PLY文件
        pcd = o3d.io.read_point_cloud(f"{pcd_path}/{file}")
        
        # 下采样
        # pcd = pcd.voxel_down_sample(voxel_size=0.05)
        
        # 保存下采样后的PLY文件
        o3d.io.write_point_cloud(f"/home/ubuntu/DATA1/junhaoge/sg_assets/agent_ply/{file}", pcd, write_ascii=True)