import numpy as np
import open3d as o3d
import os


def sort_by_number(file_name):
    Initial = int(os.path.splitext(file_name)[0].split("_")[1])
    return Initial


def add_color(pc):
    color_list = np.loadtxt(color_list_path)
    label = pc[:, -1].reshape(-1, 1)
    pc_color = np.zeros((label.shape[0], 3))
    lab = np.unique(label)
    for l in range(len(lab)):
        index = np.where(label == lab[l])[0]
        pc_color[index, :] = color_list[int(lab[l]), :]
    return pc_color


def show_pointcloud_dir(file_path):
    g_idx = 0

    files = sorted(os.listdir(file_path),key=sort_by_number)
    filelist = [os.path.join(file_path, f) for f in files]
    vis = o3d.visualization.VisualizerWithKeyCallback()

    def show_pointcloud(vis):
        nonlocal g_idx
        points = np.genfromtxt(filelist[g_idx], delimiter=" ")
        print(filelist[g_idx])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        color = add_color(points)
        pcd.colors = o3d.utility.Vector3dVector(color)

        vis.clear_geometries()
        vis.add_geometry(pcd)
        vis.update_renderer()
        vis.poll_events()

    def key_forward_callback(vis):
        nonlocal g_idx
        g_idx += 1
        if g_idx >= len(filelist):
            g_idx = len(filelist) - 1
        show_pointcloud(vis)
        return True

    def key_back_callback(vis):
        nonlocal g_idx
        g_idx -= 1
        if g_idx < 0:
            g_idx = 0
        show_pointcloud(vis)
        return True

    vis.create_window(width=1600, height=1600, left=1000, top=200)
    vis.get_render_option().point_size = 5
    vis.register_key_callback(ord('D'), key_forward_callback)
    vis.register_key_callback(ord('A'), key_back_callback)
    vis.get_view_control().change_field_of_view(step=-90)
    vis.run()

if __name__ == "__main__":
    home_path = os.path.dirname(os.path.dirname(os.getcwd()))
    color_list_path = os.path.join(home_path, "color_list.txt")
    path = os.path.join(home_path, "instance_old_organ(by human)")
    show_pointcloud_dir(path)