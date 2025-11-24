import numpy as np
import os
import torch
import json
import pickle
import open3d as o3d

#from src.chamfer import *
from src.visualisation import visualize_predictions
from src.chamfer import *

def scale_preds(preds, cat, up=1, norm_factor=1, scale_positions=False):
    if up == 1:
        scale_factor = 1000
    elif up == 0:
        scale_factor = 0.001
    else:
        scale_factor = 1

    if cat == "pipe":
        if scale_positions:
            scalable_targets = [0, 1, 2, 3, 4]
        else:
            scalable_targets = [0, 1]

    elif cat == "flange":
        if scale_positions:
            scalable_targets = [0, 1, 2, 3, 4, 5, 6]
        else:
            scalable_targets = [0, 1, 2, 3]

    elif cat == "elbow" or cat == "bend":
        if scale_positions:
            scalable_targets = [0, 1, 2, 3, 4, 5]
        else:
            scalable_targets = [0, 1, 2]

    elif cat == "tee":
        if scale_positions:
            scalable_targets = [0, 1, 2, 3, 4, 5, 6]
        else:
            scalable_targets = [0, 1, 2, 3]

    for j in scalable_targets:
        preds[j] = preds[j] * scale_factor * norm_factor

    # handle negative radius
    # preds[0] = abs(preds[0])

    return preds


def translate_preds(preds, cat, translation):
    if cat == "tee" or cat == "flange":
        targets = [4, 5, 6]
    elif cat == "elbow" or cat == "bend":
        targets = [3, 4, 5]
    elif cat == "pipe":
        targets = [2, 3, 4]

    for i, t in enumerate(targets):
        preds[t] = preds[t] + translation[i]
    return preds


def prepare_visualisation(
    pcd_id, cat, cloud_id, cloud_list, predictions_list, path, ext
):
    pcd_path = path + cat + "/test/" + str(pcd_id) + ext

    # load pcd and 'un-normalise'
    pcd_temp = o3d.io.read_point_cloud(pcd_path).points
    norm_pcd_temp = np.mean(pcd_temp, axis=0)
    pcd_temp -= norm_pcd_temp
    norm_factor = np.max(np.linalg.norm((pcd_temp), axis=1))
    points = (cloud_list[cloud_id].transpose(1, 0)) * norm_factor  # + norm_pcd_temp
    pcd = o3d.utility.Vector3dVector(points)

    # scale predictions when necessary
    preds = predictions_list[cloud_id].tolist()
    preds = scale_preds(preds, cat, norm_factor=norm_factor)

    preds = [float(pred) for pred in preds]
    return pcd, preds


def undo_normalisation(pcd_id, cat, preds, path, ext, scale_up=False):
    pcd_path = path / (cat + "/test/" + str(pcd_id) + ext)

    # load pcd and 'un-normalise'
    pcd_temp = o3d.io.read_point_cloud(str(pcd_path)).points
    norm_pcd_temp = np.mean(pcd_temp, axis=0)
    pcd_temp -= norm_pcd_temp
    norm_factor = np.max(np.linalg.norm((pcd_temp), axis=1))

    # scale and translate predictions
    up = 1 if scale_up else 2
    preds = scale_preds(
        preds, cat, up=up, norm_factor=norm_factor, scale_positions=True
    )
    preds = translate_preds(preds, cat, norm_pcd_temp)
    preds = [float(pred) for pred in preds]
    return preds


def load_preds(preds_dir, cat):
    preds_file = preds_dir / ("preds_finetuned_" + cat + ".pkl")
    with open(preds_file, "rb") as f:
        return pickle.load(f)


def bp_tee_correction(original_pred, cloud_data, cat):
    preds = scale_preds(
        original_pred,
        cat,
        up=2,
        norm_factor=cloud_data["norm_factor"],
        scale_positions=True,
    )
    preds = translate_preds(preds, cat, cloud_data["mean"])
    return preds


# visualise all predictions together
def batch_visualise(preds_dir, blueprint, path, ext, device, ifc=True):
    # load predictions
    all_dists = []
    for cat in ["tee", "elbow", "pipe", "flange"]:
        # for cat in ['pipe']:
        preds, ids, dists = load_preds(preds_dir, cat)
        print(cat, len(preds))
        all_dists.append(dists)

        if cat == "tee":
            metadata_file = open(path / (cat + "/metadata.json"), "r")
            metadata = json.load(metadata_file)

        original_preds = []
        for i in range(len(preds)):
            pcd_id = ids[i].item()
            original_pred = undo_normalisation(
                pcd_id, cat, preds[i], path, ext, scale_up=ifc
            )
            # tees require an additional level of normalisation since the dataset was
            # resampled to avoid issues with capped ends
            if cat == "tee":
                original_pred = bp_tee_correction(
                    original_pred, metadata[str(pcd_id)], cat
                )

            original_preds.append(original_pred)

        # generate ifc models for predictions
        if ifc:
            ifc_file = visualize_predictions(
                None, cat, original_preds, blueprint, visualize=False
            )
            ifc_file.write(str(preds_dir / (cat + "_bp.ifc")))

        else:
            # generate point clouds for predictions
            preds_tensor = torch.Tensor(original_preds).to(device)

            if cat == "elbow":
                target_pcd_tensor = generate_elbow_cloud_tensor(preds_tensor)
            elif cat == "pipe":
                target_pcd_tensor = generate_pipe_cloud_tensor(preds_tensor)
            elif cat == "flange":
                target_pcd_tensor = generate_flange_cloud_tensor(preds_tensor)
            elif cat == "tee":
                # target_pcd_tensor = generate_tee_cloud_tensor(preds_tensor)
                # temporary fix using cpu code to fix point deletion error
                points = []
                for pred in original_preds:
                    points.append(generate_tee_cloud(np.array(pred)))
                points = np.concatenate(points)

            if cat != "tee":
                points = target_pcd_tensor.cpu().detach().numpy()
                points = np.reshape(
                    points, (points.shape[0] * points.shape[1], points.shape[2])
                )
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(str(preds_dir / (cat + "_bp.pcd")), cloud)


def merge_clouds(directory, cat):
    path = str(directory / (cat + "/test/"))
    files = os.listdir(path)
    points_lists = []
    for f in files:
        if f.split(".")[-1] == "pcd" or f.split(".")[-1] == "ply":
            pcd_path = path + "/" + f
            points_lists.append(o3d.io.read_point_cloud(pcd_path).points)

    all_points = np.concatenate(points_lists)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(all_points)
    o3d.io.write_point_cloud(cat + "_bp_input.pcd", cloud)
