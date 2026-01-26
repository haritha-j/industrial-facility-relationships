import numpy as np
import math
import os
import json
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

from src.preparation import *


def parse_tee_properties(element_data, use_directions=True):
    # target = [element_data['radius']/1000, element_data['length']/1000]
    scaled_targets = [
        element_data["radius1"] / 1000,
        element_data["length1"] / 1000,
        element_data["radius2"] / 1000,
        element_data["length2"] / 1000,
    ]
    unscaled_targets = []
    position_targets = [
        element_data["position2"][0] / 1000,
        element_data["position2"][1] / 1000,
        element_data["position2"][2] / 1000,
    ]

    if use_directions:
        for d in ["direction1", "direction2"]:
            for i in range(3):
                unscaled_targets.append(math.sin(element_data[d][i]))
                unscaled_targets.append(math.cos(element_data[d][i]))
    else:
        p2 = (
            np.array(element_data["position1"]) / 1000
            + (
                np.array(element_data["direction1"])
                * np.array(element_data["length1"] / 1000 * 0.5)
            )
        ).tolist()
        for i in range(3):
            position_targets.append(p2[i])
        p3 = (
            np.array(p2)
            + (
                np.array(element_data["direction2"])
                * np.array(element_data["length2"] / 1000)
            )
        ).tolist()
        for i in range(3):
            position_targets.append(p3[i])

    # target = [element_data['radius']/1000]
    return (
        np.array(scaled_targets),
        np.array(unscaled_targets),
        np.array(position_targets),
    )


def parse_pipe_properties(element_data):
    scaled_targets = [element_data["radius"] / 1000, element_data["length"] / 1000]
    unscaled_targets = []

    for i in range(3):
        unscaled_targets.append(math.sin(element_data["direction"][i]))
        unscaled_targets.append(math.cos(element_data["direction"][i]))

    position_targets = [0.0, 0.0, 0.0]

    return (
        np.array(scaled_targets),
        np.array(unscaled_targets),
        np.array(position_targets),
    )


def parse_flange_properties(element_data):
    scaled_targets = [
        element_data["radius1"] / 1000,
        element_data["radius2"] / 1000,
        element_data["length1"] / 1000,
        element_data["length2"] / 1000,
    ]
    unscaled_targets = []

    for i in range(3):
        unscaled_targets.append(math.sin(element_data["direction"][i]))
        unscaled_targets.append(math.cos(element_data["direction"][i]))

    position_targets = [
        element_data["position"][0] / 1000,
        element_data["position"][1] / 1000,
        element_data["position"][2] / 1000,
    ]

    return (
        np.array(scaled_targets),
        np.array(unscaled_targets),
        np.array(position_targets),
    )


def parse_elbow_properties(element_data):
    # target = [element_data['radius']/1000, element_data['length']/1000]
    scaled_targets = [
        element_data["radius"] / 1000,
        element_data["axis_x"] / 1000,
        element_data["axis_y"] / 1000,
    ]
    unscaled_targets = [
        math.sin(math.radians(element_data["angle"])),
        math.cos(math.radians(element_data["angle"])),
    ]
    # target = [element_data['radius']/1000]
    position_targets = [
        element_data["position"][0] / 1000,
        element_data["position"][1] / 1000,
        element_data["position"][2] / 1000,
    ]

    for i in range(3):
        unscaled_targets.append(math.sin(element_data["direction"][i]))
        unscaled_targets.append(math.cos(element_data["direction"][i]))

    return (
        np.array(scaled_targets),
        np.array(unscaled_targets),
        np.array(position_targets),
    )


def parse_ibeam_properties(element_data):
    scaled_targets = [
        element_data["width"] / 1000,
        element_data["depth"] / 1000,
        element_data["web_thickness"] / 1000,
        element_data["flange_thickness"] / 1000,
        element_data["fillet_radius"] / 1000,
        element_data["length"] / 1000,
    ]
    unscaled_targets = []

    for i in range(3):
        unscaled_targets.append(math.sin(element_data["direction"][i]))
        unscaled_targets.append(math.cos(element_data["direction"][i]))

    position_targets = [
        element_data["position"][0] / 1000,
        element_data["position"][1] / 1000,
        element_data["position"][2] / 1000,
    ]

    return (
        np.array(scaled_targets),
        np.array(unscaled_targets),
        np.array(position_targets),
    )


def parse_lbeam_properties(element_data):
    scaled_targets = [
        element_data["width"] / 1000,
        element_data["depth"] / 1000,
        element_data["thickness"] / 1000,
        element_data["fillet_radius"] / 1000,
        element_data["length"] / 1000,
    ]
    unscaled_targets = []

    for i in range(3):
        unscaled_targets.append(math.sin(element_data["direction"][i]))
        unscaled_targets.append(math.cos(element_data["direction"][i]))

    position_targets = [
        element_data["position"][0] / 1000,
        element_data["position"][1] / 1000,
        element_data["position"][2] / 1000,
    ]

    return (
        np.array(scaled_targets),
        np.array(unscaled_targets),
        np.array(position_targets),
    )


def parse_cbeam_properties(element_data):
    scaled_targets = [
        element_data["width"] / 1000,
        element_data["depth"] / 1000,
        element_data["wall_thickness"] / 1000,
        element_data["girth"] / 1000,
        element_data["fillet_radius"] / 1000,
        element_data["length"] / 1000,
    ]
    unscaled_targets = []

    for i in range(3):
        unscaled_targets.append(math.sin(element_data["direction"][i]))
        unscaled_targets.append(math.cos(element_data["direction"][i]))

    position_targets = [
        element_data["position"][0] / 1000,
        element_data["position"][1] / 1000,
        element_data["position"][2] / 1000,
    ]

    return (
        np.array(scaled_targets),
        np.array(unscaled_targets),
        np.array(position_targets),
    )


def default_transforms():
    return transforms.Compose([Normalize(), ToTensor()])


# scaled properties must be transformed when the cloud's scale is transformed
class PointCloudData(Dataset):
    def __init__(
        self,
        root_dir,
        valid=False,
        folder="train",
        category="pipe",
        transform=default_transforms(),
        inference=False,
        return_coefficients=False,
    ):
        self.root_dir = root_dir
        folders = [
            dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir / dir)
        ]
        self.category = category
        self.transforms = transform if not valid else default_transforms()
        if not inference:
            metadata_file = open(root_dir / Path(category) / "metadata_new.json", "r")
            metadata = json.load(metadata_file)
        self.valid = valid
        self.inference = inference
        self.return_coefficients = return_coefficients
        self.files = []

        new_dir = root_dir / Path(category) / folder
        for file in os.listdir(new_dir):
            if file.endswith(".pcd") or file.endswith(".ply"):
                sample = {}
                sample["pcd_path"] = new_dir / file
                try:
                    sample["id"] = int(file.split(".")[0])
                except ValueError:
                    sample["id"] = file.split(".")[0]
                if not self.inference:
                    if category == "pipe":
                        (
                            sample["scaled_properties"],
                            sample["unscaled_properties"],
                            sample["position_properties"],
                        ) = parse_pipe_properties(metadata[file.split(".")[0]])
                    elif category == "elbow":
                        (
                            sample["scaled_properties"],
                            sample["unscaled_properties"],
                            sample["position_properties"],
                        ) = parse_elbow_properties(metadata[file.split(".")[0]])
                    elif category == "flange":
                        (
                            sample["scaled_properties"],
                            sample["unscaled_properties"],
                            sample["position_properties"],
                        ) = parse_flange_properties(metadata[file.split(".")[0]])
                    elif category == "tee":
                        (
                            sample["scaled_properties"],
                            sample["unscaled_properties"],
                            sample["position_properties"],
                        ) = parse_tee_properties(metadata[file.split(".")[0]])
                    elif category == "ibeam":
                        (
                            sample["scaled_properties"],
                            sample["unscaled_properties"],
                            sample["position_properties"],
                        ) = parse_ibeam_properties(metadata[file.split(".")[0]])
                    elif category == "lbeam":
                        (
                            sample["scaled_properties"],
                            sample["unscaled_properties"],
                            sample["position_properties"],
                        ) = parse_lbeam_properties(metadata[file.split(".")[0]])
                    elif category == "cbeam":
                        (
                            sample["scaled_properties"],
                            sample["unscaled_properties"],
                            sample["position_properties"],
                        ) = parse_cbeam_properties(metadata[file.split(".")[0]])
                self.files.append(sample)

        if not inference:
            self.targets = (
                len(self.files[0]["scaled_properties"])
                + len(self.files[0]["unscaled_properties"])
                + len(self.files[0]["position_properties"])
            )

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file, scaled_properties, position_properties):
        cloud = read_pcd(file)
        if self.transforms:
            cloud, scaled_properties, position_properties, mean, norm_factor = self.transforms(
                (cloud, scaled_properties, position_properties)
            )
        if self.return_coefficients:
            return cloud, scaled_properties, position_properties,  mean, norm_factor
        return cloud, scaled_properties, position_properties

    def __getitem__(self, idx):
        pcd_path = self.files[idx]["pcd_path"]
        id = self.files[idx]["id"]

        if not self.inference:
            scaled_properties = self.files[idx]["scaled_properties"]
            position_properties = self.files[idx]["position_properties"]
            unscaled_properties = torch.from_numpy(
                self.files[idx]["unscaled_properties"]
            ).float()
            pointcloud, scaled_properties, position_properties = self.__preproc__(
                pcd_path, scaled_properties, position_properties
            )

            return {
                "pointcloud": pointcloud,
                "properties": torch.cat(
                    (scaled_properties, position_properties, unscaled_properties)
                ),
                "id": id,
            }
        else:
            if self.return_coefficients:
                pointcloud, _, _, mean, norm_factor = self.__preproc__(pcd_path, np.ones(10), np.ones(3))
                return {"pointcloud": pointcloud, "id": id, "mean": mean, "norm_factor": norm_factor}
            else:
                pointcloud, _, _ = self.__preproc__(pcd_path, np.ones(10), np.ones(3))
                return {"pointcloud": pointcloud, "id": id}
