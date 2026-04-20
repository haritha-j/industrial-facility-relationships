
import numpy as np
import os
import os.path
import torch
import pickle

import matplotlib.pyplot as plt
from chamferdist import ChamferDistance
import open3d as o3d

from src.visualisation import *



# get densities
def get_density(clouds):
    # compute nearest neighbours to calculated density
    clouds = torch.tensor(clouds, device="cuda")
    chamferDist = ChamferDistance()
    nn = chamferDist(clouds, clouds, bidirectional=False, return_nn=True, k=32)

    density = torch.mean(nn[0].dists[:,:,1:], dim=2)
    eps = 0.00001
    density = 1 / (density + eps)
    return density


# normalise for each example across prediction sets
def normalise_densities(density_sets):
    densities = torch.stack(density_sets)
    highs, lows = torch.max(densities, 2).values, torch.min(densities, 2).values
    highs, lows = torch.max(highs, 0).values, torch.min(lows, 0).values
    #print(densities.shape, highs.shape)

    #highs = torch.reshape(highs, densities.shape)
    highs = highs.unsqueeze(0).unsqueeze(-1)
    highs = highs.expand(densities.shape[0], densities.shape[1], densities.shape[2])
    lows = lows.unsqueeze(0).unsqueeze(-1)
    lows = lows.expand(densities.shape[0], densities.shape[1], densities.shape[2])
    diff = highs - lows
    densities = (densities - lows) / diff

    return densities[0], densities[1], densities[2], densities[3]



# produce density colourmaps that are normalised with their pairs
# represent density with colour and combine with point cloud
def get_coloured_clouds(clouds, density, colormap_name='plasma_r'):
    try:
        density = density.detach().cpu().numpy()
    except:
        pass
    colours = np.zeros((density.shape[0], density.shape[1], 4))
    colormap = plt.get_cmap(colormap_name)

    for i, cloud in tqdm(enumerate(density)):
        for j, pt in enumerate(cloud):
            colours[i,j] = colormap(pt)

#     clouds = clouds.detach().cpu().numpy()
    colours = colours[:,:,:3]
    pcds = []

    for i, cl in enumerate(clouds):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cl)
        pcd.colors = o3d.utility.Vector3dVector(colours[i])
        pcds.append(pcd)

    return pcds, colours


def get_cloud_list_vcn(path, prefix, pred_bbox=False, pred_camera=False, pred_confidence=False, limit=100, load_idx = False):
    cloud_sets = []

    for i in range(limit):
        f_name = "fine"+str(i)+".pkl"
        with open(path + f_name, "rb") as f:
            if pred_bbox:
                pred, partial, gt, predicted_bbox = pickle.load(f)
                if prefix == "pred":
                    clouds = pred
                elif prefix == "gt":
                    clouds = gt
                elif prefix == "pred_bbox":
                    clouds = predicted_bbox
                else:
                    clouds = partial
            if pred_camera:
                pred, partial, gt, gt_camera, predicted_camera = pickle.load(f)
                if prefix == "pred":
                    clouds = pred
                elif prefix == "gt":
                    clouds = gt
                elif prefix == "pred_camera":
                    clouds = predicted_camera
                elif prefix == "gt_camera":
                    clouds = gt_camera
                else:
                    clouds = partial
            elif pred_confidence:
                if load_idx:
                    pred, partial, gt, confidence, idx = pickle.load(f)
                else:
                    pred, partial, gt, confidence = pickle.load(f)
                if prefix == "pred":
                    clouds = pred
                elif prefix == "gt":
                    clouds = gt
                elif prefix == "confidence":
                    clouds = confidence
                elif prefix == "idx":
                        clouds = idx
                else:
                    clouds = partial
            else:
                pred, partial, gt = pickle.load(f)
                if prefix == "pred":
                    clouds = pred
                elif prefix == "gt":
                    clouds = gt
                else:
                    clouds = partial
        if prefix == "idx":
            cloud_sets.append(clouds)
        else:
            cloud_sets.append(clouds.detach().cpu().numpy())
    if prefix == "idx":
        return cloud_sets
    cloud_sets = np.vstack(cloud_sets)
    print(cloud_sets.shape)

    return cloud_sets


def view_side_by_side(v1, v2, v3, v4, i):
    # shift points
    cl1, cl2, cl3, cl4 = v1[i], v2[i], v3[i], v4[i]
    cl1.points = o3d.utility.Vector3dVector(np.array(cl1.points) - np.array([1,0,0]))
    cl3.points = o3d.utility.Vector3dVector(np.array(cl3.points) + np.array([1,0,0]))
    cl4.points = o3d.utility.Vector3dVector(np.array(cl4.points) + np.array([2,0,0]))

    o3d.visualization.draw_geometries([cl1, cl2, cl3, cl4])


def view_cloud_list_vcn(path, prefix, ifc, col=0):
    limit = 2
    vis = []

    for i in range(limit):
        f_name = "fine"+str(i)+".pkl"
        with open(path + f_name, "rb") as f:
            pred, partial, gt = pickle.load(f)
            if prefix == "pred":
                clouds = pred
            elif prefix == "gt":
                clouds = gt
            else:
                clouds = partial
        for cl in clouds:
            cloud_list = [None, None, None]
            cloud_list[col] = cl.detach().cpu().numpy().astype(np.double)
            vis.append(vis_ifc_and_cloud(ifc, cloud_list))
    return vis


def view_cloud_list_seedformer(category_id, path, prefix, ifc, col=0):
    files = os.listdir(os.path.join(path,category_id))
    files_filtered = [f for f in files if prefix in f]
    files_filtered.sort()
    #print(len(files), len(files_filtered))

    limit = 10
    vis = []
    count = 0

    for f in files_filtered:
        count +=1
        if count == limit:
            break
        cloud = np.load(os.path.join(path, category_id, f))
        cloud_list = [None, None, None]
        cloud_list[col] = cloud.astype("float64")
        vis.append(vis_ifc_and_cloud(ifc, cloud_list))
    return vis


# visualise seedformer partial and output clouds
def view_seedformer_outputs(path):
    files = os.listdir(path)
    files_filtered = [f for f in files if "partial" in f]
    files_filtered.sort()
    #print(len(files), len(files_filtered))

    limit = 10
    count = 0

    for f in files_filtered:
        count +=1
        if count == limit:
            break
        prefix = f.split("_")[0]
        print(f)
        complete = np.load(os.path.join(path, prefix + "_pred.npy"))
        partial = np.load(os.path.join(path, f))
        print(len(complete), len(partial))

        complete_cloud = o3d.geometry.PointCloud()
        complete_cloud.points = o3d.utility.Vector3dVector(complete)
        partial_cloud = o3d.geometry.PointCloud()
        partial_cloud.points = o3d.utility.Vector3dVector(partial)

        o3d.io.write_point_cloud("complete_" + prefix + ".pcd", complete_cloud, write_ascii=True)

        partial_cloud.paint_uniform_color([0.0, 0.706, 1])
        complete_cloud.paint_uniform_color([0.7, 0.70, 0])

        o3d.visualization.draw_geometries([complete_cloud, partial_cloud])


# get a list of bboxes as min, max points from a set of point clouds
def get_bboxes(clouds):
    bboxes = []
    for cloud in clouds:
        min_bound = np.min(cloud, axis=0)
        max_bound = np.max(cloud, axis=0)
        bboxes.append([min_bound, max_bound])
    return np.array(bboxes)


def convert_bboxes_to_min_max(bboxes):
    center = bboxes[:,:,0]
    bounds = bboxes[:,:,1]
    min_pts = center - bounds/2
    max_pts = center + bounds/2

    bboxes_min_max = np.stack([min_pts, max_pts], axis=1)
    return bboxes_min_max


def extract_camera_location_and_direction(pose):
    # Extract the camera location (translation vector)
    location = pose[:3, 3]
    # Extract the camera direction
    rotation_matrix = pose[:3, :3]
    direction = np.dot(rotation_matrix, np.array([0, 0, 1]))

    # Normalize the direction vector to make it a unit vector
    direction = direction / np.linalg.norm(direction)
    return location, direction


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + np.dot(kmat, kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def visualize_point_cloud_with_camera_axis(point_cloud, camera_location, camera_axis):
    # Convert the point cloud to an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Normalize the camera axis for consistent visualization
    camera_axis_normalized = camera_axis / np.linalg.norm(camera_axis)
    axis_length = 0.1
    arrow_radius = 0.005
    cone_radius = 0.01
    cone_height = 0.02

    camera_axis_normalized = camera_axis_normalized*-1

    # Create an arrow to represent the camera axis
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=arrow_radius,
        cone_radius=cone_radius,
        cylinder_height=axis_length,
        cone_height=cone_height
    )

    # Align the arrow with the camera axis
    default_axis = np.array([0, 0, 1])
    rotation_matrix = rotation_matrix_from_vectors(default_axis, camera_axis_normalized)
    print(rotation_matrix)
    arrow.rotate(rotation_matrix, center=[0, 0, 0])

    # Move the arrow to the camera location
    arrow.translate(camera_location)

    # Set the color of the arrow to red
    arrow.paint_uniform_color([1, 0, 0])

    # Visualize the point cloud and camera axis
    o3d.visualization.draw_geometries([pcd, arrow])


# Add camera cones if provided
def add_camera_cone(geometries, camera, colour=[1,0,0], width=160, height=120, focal_length=100):
    camera_location = camera[:, 0]
    camera_axis = camera[:, 1]
    print(camera)

    # Normalize the camera axis for consistent visualization
    camera_axis_normalized = camera_axis / np.linalg.norm(camera_axis)

    # Calculate the half-angles of the camera's field of view
    half_angle_x = np.arctan(width / (2 * focal_length))
    half_angle_y = np.arctan(height / (2 * focal_length))

    # Use the larger of the two half-angles to define the cone's base radius
    half_angle = max(half_angle_x, half_angle_y)

    # Define the cone's dimensions
    cone_height = 0.1  # Arbitrary length of the visible region
    cone_radius = np.tan(half_angle) * cone_height

    # Create a cone to represent the camera's field of view
    cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=cone_height)

    # Align the cone with the camera axis
    default_axis = np.array([0, 0, 1])
    rotation_matrix = rotation_matrix_from_vectors(default_axis, camera_axis_normalized)
    cone.rotate(rotation_matrix, center=[0, 0, 0])

    # Move the cone to the camera location
    cone.translate(camera_location)

    # Set the color of the cone to a uniform red
    cone.paint_uniform_color(colour)

    # Create a wireframe representation of the cone
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(cone)
    wireframe.paint_uniform_color([0, 0, 0])  # Set wireframe color to black

    # Add the cone and wireframe to the geometries list
    geometries.extend([cone, wireframe])


def rgb_to_float(rgb):
    return [c / 255 for c in rgb]


def visualize_point_clouds_with_bboxes_and_cameras(pc1, pc2=None, pc3=None, bboxes1=None, bboxes2=None, bboxes3=None,
                                                   cameras1=None, cameras2=None, cameras3=None,
                                                   cameras4=None, paint_preds=True):
    """
    Display pairs of point clouds with optional bounding boxes and cameras using Open3D.

    Parameters:
    pc1 (np.ndarray): First array of point clouds with shape [b, n_points, 3].
    pc2 (np.ndarray): Second array of point clouds with shape [b, n_points, 3].
    pc3 (np.ndarray, optional): Third array of point clouds with shape [b, n_points, 3].
    bboxes1 (np.ndarray, optional): Bounding boxes for pc1 with shape [b, 2, 3].
    bboxes2 (np.ndarray, optional): Bounding boxes for pc2 with shape [b, 2, 3].
    cameras1 (np.ndarray, optional): First array of camera locations and axes with shape [b, 3, 2].
    cameras2 (np.ndarray, optional): Second array of camera locations and axes with shape [b, 3, 2].
    """
    b = len(pc1)
    #colours = [[251, 100, 10], [200, 200, 20], [2, 222, 174]] #red, yellow, green
    # red for completed bboxes, green for gt bboxes, yellow for low-confidence bboxes
    colours = [[255, 0, 0], [0, 255, 0], [255, 170, 0]]
    #colours = [[251, 100, 10], [200, 200, 20], [2, 222, 174]]
    #colours = [[202, 168, 245], [251, 86, 7], [2, 174, 174]]
    colours = [rgb_to_float(colour) for colour in colours]

    for i in range(b):
        # Create Open3D point clouds
        if paint_preds:
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(pc1[i])
            pcd1.paint_uniform_color(colours[0])  # Red color for the first point cloud
        else:
            pcd1 = pc1[i]

        geometries = [pcd1]

        if pc2 is not None:
            geometries = [pcd1]
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(pc2[i])
            pcd2.paint_uniform_color(colours[1])  # Blue color for the second point cloud
            geometries.append(pcd2)

        if pc3 is not None:
            pcd3 = o3d.geometry.PointCloud()
            pcd3.points = o3d.utility.Vector3dVector(pc3[i])
            pcd3.paint_uniform_color(colours[2])  # Green color for the third point cloud
            geometries.append(pcd3)

        # Add bounding boxes if provided
        if bboxes1 is not None:
            bbox1_min = bboxes1[i, 0]
            bbox1_max = bboxes1[i, 1]
            bbox1 = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox1_min, max_bound=bbox1_max)
            bbox1.color = colours[0] # Red color for bbox1
            geometries.append(bbox1)

        if bboxes2 is not None:
            bbox2_min = bboxes2[i, 0]
            bbox2_max = bboxes2[i, 1]
            bbox2 = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox2_min, max_bound=bbox2_max)
            bbox2.color = colours[1]  # Green color for bbox2
            geometries.append(bbox2)

        if bboxes3 is not None:
            bbox3_min = bboxes3[i, 0]
            bbox3_max = bboxes3[i, 1]
            bbox3 = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox3_min, max_bound=bbox3_max)
            bbox3.color = colours[2]  # blue color for bbox3
            geometries.append(bbox3)


        if cameras1 is not None:
            add_camera_cone(geometries, cameras1[i], colour=rgb_to_float([239, 71, 111]))
        if cameras2 is not None:
            add_camera_cone(geometries, cameras2[i], colour=rgb_to_float([255, 209, 102]))
        if cameras3 is not None:
            add_camera_cone(geometries, cameras3[i], colour=rgb_to_float([17, 138, 178]))
        if cameras4 is not None:
            add_camera_cone(geometries, cameras4[i], colour=rgb_to_float([6, 214, 160]))

        # Visualize
        o3d.visualization.draw_geometries(geometries)


def load_kitti_results_pointattn(path, prefix):
    limit = 20
    cloud_sets = []

    for i in range(limit):
        f_name = str(i)+".pkl"
        with open(path + f_name, "rb") as f:
            pred, partial, bbox = pickle.load(f)
            if prefix == "pred":
                clouds = pred
            elif prefix == "bbox":
                clouds = bbox
            else:
                clouds = partial
        cloud_sets.append(clouds.detach().cpu().numpy())
    cloud_sets = np.vstack(cloud_sets)
    print(cloud_sets.shape)

    return cloud_sets
