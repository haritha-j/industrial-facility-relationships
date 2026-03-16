# viusalize elements / relationships

import math
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm.notebook import tqdm
import matplotlib.colors as mcolors
import numpy as np

from OCC.Core.gp import gp_Pnt
from utils.JupyterIFCRenderer import JupyterIFCRenderer
import plotly.graph_objects as go
import plotly.express as px
from chamferdist import ChamferDistance
from pythreejs import (
    LineBasicMaterial,
    LineSegments2,
    LineSegmentsGeometry,
    LineMaterial,
)
from chamferdist import ChamferDistance

from src.geometry import *
from src.elements import *
from src.chamfer import *


# covert rgb to hex value
def rgb_to_hex(r, g, b):
    # Ensure that the RGB values are within the valid range (0-255)
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    # Convert RGB values to a hexadecimal color code
    hex_color = "#{:02X}{:02X}{:02X}".format(r, g, b)
    return hex_color


# visualize ifc model and point cloud simultaneously
def vis_ifc_and_cloud(ifc, clouds):
    viewer = JupyterIFCRenderer(ifc, size=(400, 300))
    colours = ["#ff7070", "#70ff70", "#7070ff"]
    for i, cloud in enumerate(clouds):
        if cloud is not None:
            gp_pnt_list = [gp_Pnt(k[0], k[1], k[2]) for k in cloud]
            # print("no points:", len(gp_pnt_list))
            col = i if i < len(colours) else 0
            viewer.DisplayShape(gp_pnt_list, vertex_color=colours[col])
    return viewer


# recover axis direction from 2 position values starting from index k, j
def get_direction_from_position(preds, k, j):
    dir = [
        (preds[k] - preds[j]),
        (preds[k + 1] - preds[j + 1]),
        (preds[k + 2] - preds[j + 2]),
    ]
    return vector_normalise(dir)


# visualize predictions side by side with ifc
def visualize_predictions(
    clouds,
    element,
    preds_list,
    blueprint,
    use_directions=True,
    visualize=True,
    z=(0.0, 0.0, 1.0),
):
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

    for preds in preds_list:
        if element == "pipe":
            pm = {"r": preds[0], "l": preds[1]}
            pm["d"] = get_direction_from_trig(preds, 5)
            pm["p0"] = [preds[2] * 1000, preds[3] * 1000, preds[4] * 1000]
            pm["p"] = [pm["p0"][i] - ((pm["l"] * pm["d"][i]) / 2) for i in range(3)]
            # print(pm)

            create_IfcPipe(pm["r"], pm["l"], pm["d"], pm["p"], ifc, ifc_info)

        elif element == "flange":
            pm = {"r1": preds[0], "r2": preds[1], "l1": preds[2], "l2": preds[3]}
            pm["d"] = get_direction_from_trig(preds, 7)
            pm["p0"] = [preds[4] * 1000, preds[5] * 1000, preds[6] * 1000]
            pm["p"] = [pm["p0"][i] - ((pm["l1"] * pm["d"][i])) for i in range(3)]
            # print(pm)

            create_IfcFlange(
                pm["r1"],
                pm["r2"],
                pm["l1"],
                pm["l2"],
                pm["d"],
                pm["p"],
                pm["p0"],
                ifc,
                ifc_info,
            )

        elif element == "elbow":
            pm = {"r": preds[0], "x": preds[1], "y": preds[2]}

            theta = math.atan2(pm["x"], pm["y"])
            pm["axis_dir"] = [math.cos(theta), -1 * math.sin(theta)]
            # pm['p'] = [0.0, 0.0, 0.0]
            pm["a"] = math.degrees(math.atan2(preds[6], preds[7]))

            pm["p"] = [preds[3] * 1000, preds[4] * 1000, preds[5] * 1000]
            pm["d"] = get_direction_from_trig(preds, 8)

            # print("all", pm['r'], pm['a'], pm['d'], pm['p'], pm['x'],
            #                 pm['y'], pm['axis_dir'])

            create_IfcElbow(
                pm["r"],
                pm["a"],
                pm["d"],
                pm["p"],
                pm["x"],
                pm["y"],
                pm["axis_dir"],
                ifc,
                ifc_info,
                z=z,
            )

        elif element == "tee":
            pm = {"r1": preds[0], "l1": preds[1], "r2": preds[2], "l2": preds[3]}
            pm["p2"] = [preds[4] * 1000, preds[5] * 1000, preds[6] * 1000]

            if use_directions:
                pm["d1"] = get_direction_from_trig(preds, 7)
                pm["d2"] = get_direction_from_trig(preds, 13)
                pm["p1"] = (
                    np.array(pm["p2"]) - (np.array(pm["d1"]) * np.array(pm["l1"]) * 0.5)
                ).tolist()
            else:
                pm["d1"] = get_direction_from_position(preds, 7, 4)
                pm["d2"] = get_direction_from_position(preds, 10, 7)
                pm["p2"] = [preds[7] * 1000, preds[8] * 1000, preds[9] * 1000]

            create_IfcTee(
                pm["r1"],
                pm["r2"],
                pm["l1"],
                pm["l2"],
                pm["d1"],
                pm["d2"],
                pm["p1"],
                pm["p2"],
                ifc,
                ifc_info,
            )

        elif element == "ibeam":
            pm = {
                "width": preds[0],
                "depth": preds[1],
                "web_thickness": preds[2],
                "flange_thickness": preds[3],
                "fillet_radius": preds[4],
                "length": preds[5],
            }
            pm["d"] = get_direction_from_trig(preds, 9)
            pm["p"] = [preds[6] * 1000, preds[7] * 1000, preds[8] * 1000]

            create_IfcIBeam(
                pm["width"],
                pm["depth"],
                pm["web_thickness"],
                pm["flange_thickness"],
                pm["fillet_radius"],
                pm["length"],
                pm["d"],
                pm["p"],
                ifc,
                ifc_info,
            )

        elif element == "lbeam":
            pm = {
                "width": preds[0],
                "depth": preds[1],
                "thickness": preds[2],
                "fillet_radius": preds[3],
                "length": preds[4],
            }
            pm["d"] = get_direction_from_trig(preds, 8)
            pm["p"] = [preds[5] * 1000, preds[6] * 1000, preds[7] * 1000]

            create_IfcLBeam(
                pm["width"],
                pm["depth"],
                pm["thickness"],
                pm["thickness"],
                pm["fillet_radius"],
                pm["length"],
                pm["d"],
                pm["p"],
                ifc,
                ifc_info,
            )

        elif element == "cbeam":
            pm = {
                "width": preds[0],
                "depth": preds[1],
                "wall_thickness": preds[2],
                "girth": preds[3],
                "fillet_radius": preds[4],
                "length": preds[5],
            }
            pm["d"] = get_direction_from_trig(preds, 9)
            pm["p"] = [preds[6] * 1000, preds[7] * 1000, preds[8] * 1000]

            create_IfcCBeam(
                pm["width"],
                pm["depth"],
                pm["wall_thickness"],
                pm["girth"],
                pm["fillet_radius"],
                pm["length"],
                pm["d"],
                pm["p"],
                ifc,
                ifc_info,
            )

    # ifc.write("temp.ifc")
    if visualize:
        return vis_ifc_and_cloud(ifc, clouds), ifc
    else:
        return ifc, None


def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames = []

    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(
            dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze)))))
        )
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1,
                    x=0.8,
                    xanchor="left",
                    yanchor="bottom",
                    pad=dict(t=45, r=10),
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=50, redraw=True),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                    mode="immediate",
                                ),
                            ],
                        )
                    ],
                )
            ]
        ),
        frames=frames,
    )

    return fig


def pcshow(xs, ys, zs):
    data = [go.Scatter3d(x=xs, y=ys, z=zs, mode="markers")]
    fig = visualize_rotate(data)
    fig.update_traces(
        marker=dict(size=2, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )
    fig.show()


# visualise a pair of src and tgt points based on their predicted parameters
# currently expects a 1:1mapping between src and tgt points.
def visualise_parameter_pair(src_preds, tgt_preds, cat, blueprint, idx=0):
    from src.utils import scale_preds

    # prepare data on gpu and setup optimiser
    cuda = torch.device("cuda")

    preds_copy = copy.deepcopy(src_preds)
    scaled_original_preds = [scale_preds(pc.tolist(), cat) for pc in preds_copy]

    src_preds_t = torch.tensor(src_preds, requires_grad=True, device=cuda)
    tgt_preds_t = torch.tensor(tgt_preds, requires_grad=True, device=cuda)

    # generate clouds
    if cat == "elbow":
        target_pcd_tensor = generate_elbow_cloud_tensor(tgt_preds_t)
        src_pcd_tensor = generate_elbow_cloud_tensor(src_preds_t)
    elif cat == "pipe":
        target_pcd_tensor = generate_pipe_cloud_tensor(tgt_preds_t)
        src_pcd_tensor = generate_pipe_cloud_tensor(src_preds_t)
    elif cat == "tee":
        target_pcd_tensor = generate_tee_cloud_tensor(tgt_preds_t, bp=True)
        src_pcd_tensor = generate_tee_cloud_tensor(src_preds_t, bp=True)
    elif cat == "flange":
        target_pcd_tensor = generate_flange_cloud_tensor(tgt_preds_t, disc=True)
        src_pcd_tensor = generate_flange_cloud_tensor(src_preds_t, disc=True)

    # chamferDist = ChamferDistance()
    # bidirectional_nn = chamferDist(
    #     src_pcd_tensor, target_pcd_tensor, bidirectional=True, return_nn=True
    # )

    tgt_pcd = target_pcd_tensor.detach().cpu().numpy()
    src_pcd = src_pcd_tensor.detach().cpu().numpy()

    tgt_pcd_single, src_pcd_single = tgt_pcd[idx], src_pcd[idx]

    # generate ifc model and visualisation
    return visualise_matching_points(src_pcd_single, tgt_pcd_single, blueprint)


# add cloud to visualisation
def add_cloud(v, cloud, colour="#70ff70"):
    gp_pnt_list = [gp_Pnt(k[0], k[1], k[2]) for k in cloud]
    v.DisplayShape(gp_pnt_list, vertex_color=colour)


# draw lines connecting src and tgt points
def add_lines(v, src, tgt, pairs=None, k=1):
    lines = []
    if pairs is None: pairs = [i for i in range(len(src))]
    if k==1: pairs = [pairs]

    for j in range(k):
        positions = [[src[i], tgt[pairs[j][i]]] for i in range(len(tgt))]
        print(len(positions))
        lines.append(
            LineSegments2(
                LineSegmentsGeometry(positions=positions),
                LineMaterial(linewidth=2, color="green"),
            )
        )

    v._displayed_non_pickable_objects.add(lines)


# draw lines connecting src and tgt points
# strength is used to determine colour intensity.
# this function is much slower due to need to generate a new line segment for each pair
def add_lines_colour(v, src, tgt, pairs=None, k=1, strength=None):
    lines = []
    if pairs is None: pairs = [i for i in range(len(src))]
    if k==1: pairs = [pairs]
    if strength is None: strength = [1.0 for i in range(len(pairs[0]))]

    print(len(strength))
    colour_intensity = [int(255 * (1.0 - s)) for s in strength]
    colour = [rgb_to_hex(int(255 * s), int(165 * (1.0 - s) + 100 * s), 0) for s in strength]
    for i in range(len(tgt)):
        positions = [[src[i], tgt[pairs[j][i]]] for j in range(k)]

        lines.append(
            LineSegments2(
                LineSegmentsGeometry(positions=positions),
                LineMaterial(linewidth=1, color=colour[i]),
            )
        )

    v._displayed_non_pickable_objects.add(lines)

def plot_distance_colorbar(src_cld, tgt_cld, pairs=None):
    """
    Display a colour scale bar showing the distance-to-colour mapping.
    Green = small distance, Orange = large distance.
    """
    # compute distances
    if pairs is not None:
        tgt_matched = tgt_cld[pairs]
    else:
        tgt_matched = tgt_cld

    distances = np.linalg.norm(src_cld - tgt_matched, axis=1)
    #max_dist = distances.max()
    max_dist = 0.1433
    min_dist = 0

    # build matching colormap: green (0) -> orange (1)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "dist_cmap",
        [
            (0.0, (0/255, 165/255, 0/255)),    # green  (s=0, small distance)
            (1.0, (255/255, 100/255, 0/255)),   # orange (s=1, large distance)
        ]
    )

    fig, ax = plt.subplots(figsize=(5, 1))
    fig.subplots_adjust(bottom=0.5)

    norm = mcolors.Normalize(vmin=min_dist, vmax=max_dist)
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="horizontal",
    )
    cb.set_label("Euclidean distance", fontsize=12)
    cb.set_ticks([min_dist, max_dist])
    cb.set_ticklabels([f"{min_dist:.4f}", f"{max_dist:.4f}"])

    plt.title("Correspondence distance (m)", fontsize=12)
    plt.show()


# visually show matching points
# pairs is a list of indices of the matching points in tgt cloud to src cloud
def visualise_matching_points(src_cld, tgt_cld, blueprint, pairs=None, strength=None, k=1, same_cloud=False):

    # generate visualiser with blank ifc
    ifc = setup_ifc_file(blueprint)
    v = JupyterIFCRenderer(ifc, size=(700, 550))

    add_cloud(v, src_cld.astype(np.float64), colour="#0887DB")
    add_cloud(v, tgt_cld.astype(np.float64), colour="#000000")

    if same_cloud:
        tgt_cld = src_cld

    # compute strength as normalised euclidean distances if not provided
    if strength is None:
        distances = np.linalg.norm(src_cld - tgt_cld, axis=1)
        d_min, d_max = 0, 0.1433
        if d_max - d_min > 0:
            strength = (distances - d_min) / (d_max - d_min)
        else:
            strength = np.zeros(len(src_cld))

    # add elements to visualiser
    plot_distance_colorbar(src_cld, tgt_cld)
    add_lines_colour(v, src_cld, tgt_cld, pairs=pairs, strength=strength, k=k)

    return v


# generate visualisation of loss between src and tgt clouds
def visualise_loss(src_cld, tgt_cld, blueprint, loss="chamfer", strength=None, k=1, pairs=None, same_cloud=False):

    # calculate loss
    cuda = torch.device("cuda")
    target_pcd_tensor = torch.tensor([tgt_cld], device=cuda)
    src_pcd_tensor = torch.tensor([src_cld], device=cuda)

    # if loss is not chamfer, then use the pairs provided
    if loss == "chamfer":
        print("SG", src_pcd_tensor.shape, target_pcd_tensor.shape)
        chamferDist = ChamferDistance()
        nn = chamferDist(
            src_pcd_tensor, target_pcd_tensor, bidirectional=False, return_nn=True, k=k
        )

        if k == 1:
            pairs = torch.flatten(nn[0].idx[0].detach().cpu()).numpy()
            print("unique", len(np.unique(pairs)))
            #print("pairs", pairs)
        else:
            print("int", nn[0].idx[0][:,0].detach().cpu().numpy().shape)
            pairs = [nn[0].idx[0][:,i].detach().cpu().numpy() for i in range(k)]


    return visualise_matching_points(src_cld, tgt_cld, blueprint, pairs=pairs, strength=strength, k=k, same_cloud=same_cloud)



# produce a colour map based on the density of a point cloud
# parallelised version
def visualise_density(clouds, colormap_name='plasma'):
    # compute nearest neighbours to calculated density
    clouds = torch.tensor(clouds, device="cuda")
    chamferDist = ChamferDistance()
    nn = chamferDist(clouds, clouds, bidirectional=False, return_nn=True, k=32)
    print("processing density")

    # measure normalised density
    density = torch.mean(nn[0].dists[:,:,1:], dim=2)
    eps = 0.00001
    density = 1 / (density + eps)
    high, low = torch.max(density), torch.min(density)
    diff = high - low
    density = (density - low) / diff
    density = density.detach().cpu().numpy()

    # map colour
    colours = np.zeros((density.shape[0], density.shape[1], 4))
    colormap = plt.get_cmap(colormap_name)
    for i, cloud in enumerate(density):
        for j, pt in enumerate(cloud):
            colours[i,j] = colormap(pt)

    print("density map generated")
    return colours


# general function to plot multiple sets of values
def plot_dists(ax, losses, labels, title, xlabel="point cloud index", ylabel="distance (log)",
               log=True, limit=None, legend=True):
    if limit == None:
        limit = len(losses[0])
    x = np.arange(0, limit*10, 10)

    for i, loss in enumerate(losses):
        if log:
            ax.plot(x, np.log(loss[:limit]), label=labels[i])
        else:
            ax.plot(x, loss[:limit], label=labels[i])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend:
        ax.legend()