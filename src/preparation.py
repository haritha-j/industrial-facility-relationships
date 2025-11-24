import numpy as np
import math
import random
import os
import json
import torch
from tqdm.notebook import tqdm
import open3d as o3d

from src.ifc import setup_ifc_file
from src.elements import *
from src.cloud import farthest_point_sample


def read_pcd(file):
    pcd = o3d.io.read_point_cloud(str(file))
    return np.asarray(pcd.points)


class Normalize(object):
    def __call__(self, data):
        pointcloud, scaled_properties, position_properties = data[0], data[1], data[2]
        assert len(pointcloud.shape) == 2

        mean = np.mean(pointcloud, axis=0)
        norm_pointcloud = pointcloud - mean
        norm_factor = np.max(np.linalg.norm(norm_pointcloud, axis=1))
        norm_pointcloud /= norm_factor

        scaled_properties = scaled_properties / norm_factor
        position_properties = (position_properties - mean) / norm_factor
        return (norm_pointcloud, scaled_properties, position_properties, mean, norm_factor)


# trsansform the centerpoint of a cloud to origin
def center_bbox(cloud):
    bbox_max = np.amax(cloud, 0)
    bbox_min = np.amin(cloud, 0)
    print((bbox_min + bbox_max) / 2)


class RandRotation_z(object):
    def __call__(self, data):
        pointcloud, scaled_properties, position_properties = data[0], data[1], data[2]
        assert len(pointcloud.shape) == 2

        theta = random.random() * 2.0 * math.pi
        rot_matrix = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0],
                [math.sin(theta), math.cos(theta), 0],
                [0, 0, 1],
            ]
        )

        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return (rot_pointcloud, scaled_properties, position_properties)


class RandomNoise(object):
    def __call__(self, data):
        pointcloud, scaled_properties, position_properties = data[0], data[1], data[2]
        assert len(pointcloud.shape) == 2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))

        noisy_pointcloud = pointcloud + noise
        return (noisy_pointcloud, scaled_properties, position_properties)


class ToTensor(object):
    def __call__(self, data):
        return ([torch.from_numpy(i).float() if isinstance(i, np.ndarray) else i for i in data])


def random_resample_cloud(points, density, uniform_sampling):
    indices = np.arange(points.shape[0])
    if len(points) > density:
        if uniform_sampling:
            sampled_points = farthest_point_sample(points, density)
        else:
            sampled_points = points[np.random.choice(indices, density, replace=False)]
    else:
        sampled_points = points[np.random.choice(indices, density, replace=True)]

    # print(len(sampled_indices))
    return sampled_points


def save_cloud(points, output_base, name):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    save_path = os.path.join(output_base, name + ".pcd")
    o3d.io.write_point_cloud(save_path, pcd)


def synthetic_dataset(
    config, sample_size, element_class, output_base, blueprint, start=0
):
    # setup
    f = open(config, "r")
    config_data = json.load(f)
    output_dir = os.path.join(output_base, element_class)
    # os.makedirs(output_dir)

    metadata = {}
    for i in tqdm(range(start, sample_size + start)):
        # generate ifc file
        ifc = setup_ifc_file(blueprint)
        owner_history = ifc.by_type("IfcOwnerHistory")[0]
        project = ifc.by_type("IfcProject")[0]
        context = ifc.by_type("IfcGeometricRepresentationContext")[0]
        floor = ifc.by_type("IfcBuildingStorey")[0]

        ifc_info = {
            "owner_history": owner_history,
            "project": project,
            "context": context,
            "floor": floor,
        }

        # generate ifc element
        if element_class == "pipe":
            e = create_pipe(config_data[element_class], ifc, ifc_info)
        if element_class == "elbow":
            e = create_elbow(config_data[element_class], ifc, ifc_info, blueprint, i)
        elif element_class == "tee":
            e = create_tee(config_data[element_class], ifc, ifc_info, blueprint)
        elif element_class == "flange":
            e = create_flange(config_data[element_class], ifc, ifc_info, blueprint)
        elif element_class == "ibeam":
            e = create_ibeam(config_data[element_class], ifc, ifc_info)

        metadata[str(i)] = e
        ifc.write(os.path.join(output_dir, "%d.ifc" % i))

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)


def create_merged_dataset(
    pcd_path,
    output_base,
    element_class,
    num_scans,
    density,
    num_views=3,
    test_split=0.1,
    uniform_sampling=False,
    multiple=True,
):
    # load data
    metadata_file = os.path.join(output_base, element_class, "metadata.json")
    print(metadata_file)
    f = open(metadata_file, "r")
    metadata = json.load(f)

    scans = os.listdir(pcd_path)
    unique_files = set()
    for sc in scans:
        element = int(sc.split("_")[0])
        unique_files.add(element)

    if multiple:
        it_range = num_scans - num_views
    else:
        it_range = 1

    # merge multiple views
    count = 0
    metadata_new = {}
    train_clouds = {}
    test_clouds = {}
    test_point = int(len(unique_files) * (1 - test_split))
    # print(test_point, len(unique_files), sorted(unique_files))

    for k, un in enumerate(tqdm(unique_files)):
        for i in range(it_range):
            points = []
            for j in range(num_views):
                file_path = os.path.join(
                    pcd_path, (str(un) + "_" + str(i + j) + ".pcd")
                )
                pcd = o3d.io.read_point_cloud(file_path)
                points.append(pcd.points)

            # merged.points = o3d.utility.Vector3dVector(np.vstack(points))
            merged_points = np.vstack(points)
            metadata_new[str(count)] = metadata[str(un)]
            metadata_new[str(count)]["initial_ifc"] = un
            if k < test_point:
                train_clouds[str(count)] = merged_points
            else:
                test_clouds[str(count)] = merged_points

            count += 1

    # resample and save_data
    test_path = os.path.join(output_base, element_class, "test")
    train_path = os.path.join(output_base, element_class, "train")
    try:
        os.mkdir(test_path)
        os.mkdir(train_path)
    except:
        pass

    for k in tqdm(train_clouds.keys()):
        sampled_points = random_resample_cloud(
            train_clouds[k], density, uniform_sampling
        )
        save_cloud(sampled_points, train_path, k)

    for k in tqdm(test_clouds.keys()):
        sampled_points = random_resample_cloud(
            test_clouds[k], density, uniform_sampling
        )
        save_cloud(sampled_points, test_path, k)

    with open(os.path.join(output_base, element_class, "metadata_new.json"), "w") as f:
        json.dump(metadata_new, f)


def create_completion_dataset(
    pcd_path,
    output_base,
    element_class,
    num_scans,
    density,
    num_views=3,
    test_split=0.1,
    uniform_sampling=False,
    multiple=True,
    noise=False,
    noise_coverage=0.4,
    noise_factor=0.02,):

    # load data
    scans = os.listdir(pcd_path)
    unique_files = set()
    for sc in scans:
        element = int(sc.split("_")[0])
        unique_files.add(element)

    if multiple:
        it_range = num_scans - num_views
    else:
        it_range = 1

    # merge multiple views
    count = 0
    train_clouds = {}
    train_gt = {}
    test_clouds = {}
    test_gt = {}
    test_point = int(len(unique_files) * (1 - test_split))
    # print(test_point, len(unique_files), sorted(unique_files))

    for k, un in enumerate(tqdm(unique_files)):

        # create ground truth
        try:
            gt = []
            for i in range(num_scans):
                file_path = os.path.join(
                    pcd_path, (str(un) + "_" + str(i) + ".pcd")
                )
                pcd = o3d.io.read_point_cloud(file_path)
                gt.append(pcd.points)
            merged_gt = np.vstack(gt)

            # create partial clouds
            for i in range(it_range):
                points = []
                for j in range(num_views):
                    file_path = os.path.join(
                        pcd_path, (str(un) + "_" + str(i + j) + ".pcd")
                    )
                    pcd = o3d.io.read_point_cloud(file_path)
                    points.append(pcd.points)

                # merged.points = o3d.utility.Vector3dVector(np.vstack(points))
                merged_points = np.vstack(points)
                if noise:
                    noisy_points = np.random.normal(0, noise_factor, (merged_points.shape))
                    subset = np.random.choice(
                        range(merged_points.shape[0]),
                        int(merged_points.shape[0] * noise_coverage),
                        replace=False,
                    )
                    merged_points[subset] += noisy_points[subset]

                if k < test_point:
                    train_clouds[str(count)] = merged_points
                    train_gt[str(count)] = merged_gt
                else:
                    test_clouds[str(count)] = merged_points
                    test_gt[str(count)] = merged_gt
                count += 1
        except Exception as e:
            print(f"Error processing element {un}: {e}")
            continue

    # resample and save_data
    test_path = os.path.join(output_base, element_class, "test")
    train_path = os.path.join(output_base, element_class, "train")
    try:
        os.mkdir(test_path)
        os.mkdir(train_path)
    except:
        pass

    for k in tqdm(train_clouds.keys()):
        sampled_points = random_resample_cloud(
            train_clouds[k], density, uniform_sampling
        )
        save_cloud(sampled_points, train_path, k)

        sampled_points = random_resample_cloud(
            train_gt[k], density, uniform_sampling
        )
        save_cloud(sampled_points, train_path, k+"_gt")

    for k in tqdm(test_clouds.keys()):
        sampled_points = random_resample_cloud(
            test_clouds[k], density, uniform_sampling
        )
        save_cloud(sampled_points, test_path, k)

        sampled_points = random_resample_cloud(
            test_gt[k], density, uniform_sampling
        )
        save_cloud(sampled_points, test_path, k+"_gt")
