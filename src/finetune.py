# use gradient descent to fine tune parameters using chamfer loss

import time
import torch
import copy
import math
import numpy as np
from tqdm import tqdm_notebook as tqdm

from src.chamfer import *
from src.visualisation import visualize_predictions
from src.utils import scale_preds


# additional fix for elbows which have thick edges
def elbow_correction(
    n_iter, step_size, modified_preds, cloud, blueprint, gen_cloud_mod
):
    scaled_preds = [scale_preds(p.tolist(), "elbow") for p in modified_preds]
    erroneous_preds = []
    erroneous_clouds = []
    indices = []
    # filter out erreneous predictions by checking if they can be visualised
    for i, p in enumerate(tqdm(scaled_preds)):
        try:
            v_modified, _ = visualize_predictions(
                [None, None, cloud[i].transpose(1, 0).tolist()],
                "elbow",
                [scaled_preds[i]],
                blueprint,
                visualize=True,
            )
        except:
            erroneous_preds.append(modified_preds[i])
            erroneous_clouds.append(cloud[i])
            indices.append(i)

    print("err", len(erroneous_preds))

    # re-calculate errenous predictions with new restrictions
    angles = [math.pi / 2, math.pi / 4]
    x_y_pairs = [(2, 0), (0, 2), (-2, 0), (0, -2)]
    modified_erroneous_preds = []
    errors = []
    modified_clouds = []
    for ang in angles:
        for x_y in x_y_pairs:
            for j in range(len(erroneous_preds)):
                erroneous_preds[j][6] = math.sin(ang)
                erroneous_preds[j][7] = math.cos(ang)
                erroneous_preds[j][1] = erroneous_preds[j][0] * x_y[0]
                erroneous_preds[j][2] = erroneous_preds[j][0] * x_y[1]

            mp, e, mod_cld = chamfer_fine_tune(
                int(n_iter / 2),
                step_size,
                np.array(erroneous_preds).astype(np.float32),
                erroneous_clouds,
                "elbow",
                blueprint,
                alpha=2,
                visualise=False,
                elbow_fix=False,
            )
            modified_erroneous_preds.append(mp)
            modified_clouds.append(mod_cld)
            errors.append(e)
    print(len(errors))

    # pick the best prediction
    corrected_erroneous_preds = []
    corrected_mod_clouds = []
    min_errors = [math.inf for i in range(len(erroneous_preds))]
    min_indices = [0 for i in range(len(erroneous_preds))]

    for i in range(len(erroneous_preds)):
        for j in range(len(modified_erroneous_preds)):
            if errors[j][i] < min_errors[i]:
                min_errors[i] = errors[j][i]
                min_indices[i] = j

    for i in range(len(erroneous_preds)):
        corrected_erroneous_preds.append(modified_erroneous_preds[min_indices[i]][i])
        corrected_mod_clouds.append(modified_clouds[min_indices[i]][i])
    #         if errors[0][i] < errors[1][i]:
    #             print(90)
    #             corrected_erroneous_preds.append(modified_erroneous_preds[0][i])
    #             corrected_mod_clouds.append(modified_clouds[0][i])
    #         else:
    #             print(45)
    #             corrected_erroneous_preds.append(modified_erroneous_preds[1][i])
    #             corrected_mod_clouds.append(modified_clouds[1][i])

    # combine corrected preds and original preds
    count = 0
    gen_cloud_mod = list(gen_cloud_mod)
    for ind in indices:
        modified_preds[ind] = corrected_erroneous_preds[count]
        gen_cloud_mod[ind] = corrected_mod_clouds[count]

        count += 1
    print("check", count, len(corrected_erroneous_preds))

    return modified_preds, gen_cloud_mod


# multi element Adam with single loss
# set direction weight to finetune with directional chamfer loss
def chamfer_fine_tune(
    n_iter,
    step_size,
    preds,
    cloud,
    cat,
    blueprint,
    alpha=1.0,
    visualise=True,
    elbow_fix=True,
    robust=None,
    delta=0.1,
    bidirectional_robust=True,
    return_intermediate=False,
    k=1,
    direction_weight=None,
    loss_func= "chamfer",
    return_ifc=False
):
    # prepare data on gpu and setup optimiser
    cuda = torch.device("cuda")
    preds_copy = copy.deepcopy(preds)
    scaled_original_preds = [scale_preds(pc.tolist(), cat) for pc in preds_copy]


    preds_t = torch.tensor(preds, requires_grad=True, device=cuda)
    cloud_t = torch.tensor(cloud, device=cuda)
    optimiser = torch.optim.Adam([preds_t], lr=step_size)


    # check initial loss
    chamfer_loss, gen_cloud = get_chamfer_loss_tensor(
        preds_t, cloud_t, cat, reduce=False, return_cloud=True
    )
    gen_cloud = gen_cloud.detach().cpu().numpy()
    # print("intial loss", chamfer_loss)

    intermediate_results = []

    # iterative refinement with adam
    for i in tqdm(range(n_iter)):
        optimiser.zero_grad()
        if return_intermediate:
            intermediate_results.append(preds_t.clone().detach().cpu().numpy())

        if direction_weight is None:
            if loss_func == "chamfer":
                chamfer_loss = get_chamfer_loss_tensor(
                    preds_t,
                    cloud_t,
                    cat,
                    alpha=alpha,
                    robust=robust,
                    delta=delta,
                    bidirectional_robust=bidirectional_robust,
                )
            elif loss_func == "pair":
                chamfer_loss = get_pair_loss_tensor(preds_t, cloud_t, cat)
            elif loss_func == "emd":
                chamfer_loss = get_emd_loss_tensor(preds_t, cloud_t, cat)
            elif loss_func == "reverse":
                chamfer_loss = get_reverse_weighted_cd_tensor(preds_t, cloud_t, cat)
            elif loss_func == "balanced":
                chamfer_loss = get_balanced_chamfer_loss_tensor(preds_t, cloud_t, cat, k=k)
            elif loss_func == "symmetric":
                chamfer_loss = get_symmetric_chamfer_loss_tensor(preds_t, cloud_t, cat, k=k)
            elif loss_func == "infocd":
                chamfer_loss = get_infocd_loss_tensor(preds_t, cloud_t, cat)
        else:
            chamfer_loss = get_chamfer_loss_directional_tensor(
                preds_t,
                cloud_t,
                cat,
                alpha=alpha,
                robust=robust,
                delta=delta,
                bidirectional_robust=bidirectional_robust,
                k=k,
                direction_weight=direction_weight,
            )

        chamfer_loss.backward()
        optimiser.step()
        print(i, "loss", chamfer_loss.detach().cpu().numpy())  # , "preds", preds_t)

    # check final loss
    chamfer_loss, gen_cloud_mod = get_chamfer_loss_tensor(
        preds_t, cloud_t, cat, reduce=False, return_cloud=True
    )
    gen_cloud_mod = gen_cloud_mod.detach().cpu().numpy()
    #emd_loss = get_emd_loss_tensor(preds_t, cloud_t, cat)

    #print("final loss Chamfer", torch.mean(chamfer_loss).item(), "EMD", emd_loss.item())
    print("final loss Chamfer", torch.mean(chamfer_loss).item())
    modified_preds = preds_t.detach().cpu().numpy()

    # visualise
    if visualise:
        error_count = 0
        scaled_preds = [scale_preds(p.tolist(), cat) for p in modified_preds]
        visualisers = []
        cloud_visualisers = []

        for i, p in enumerate(scaled_preds):
            try:
                if not return_ifc:
                    v_orignal, _ = visualize_predictions(
                        [cloud[i].transpose(1, 0).tolist()],
                        cat,
                        [scaled_original_preds[i]],
                        blueprint,
                        visualize=(not return_ifc),
                    )
                v_modified, _ = visualize_predictions(
                    [None, None, cloud[i].transpose(1, 0).tolist()],
                    cat,
                    [scaled_preds[i]],
                    blueprint,
                    visualize=(not return_ifc),
                )
                visualisers.append(v_orignal)
                visualisers.append(v_modified)
            except:
                print("error", i)
                error_count += 1
                v_orignal, _ = visualize_predictions(
                    [cloud[i].transpose(1, 0).tolist(), gen_cloud[i].tolist()],
                    cat,
                    [],
                    blueprint,
                    visualize=(not return_ifc),
                )
                v_modified, _ = visualize_predictions(
                    [
                        None,
                        gen_cloud_mod[i].tolist(),
                        cloud[i].transpose(1, 0).tolist(),
                    ],
                    cat,
                    [],
                    blueprint,
                    visualize=(not return_ifc),
                )
                cloud_visualisers.append(v_orignal)
                cloud_visualisers.append(v_modified)

        #         return v_orignal,v_modified, modified_preds
        print("errors ", error_count)
        if return_ifc:
            return visualisers
        return cloud_visualisers, visualisers, modified_preds
    else:
        if return_intermediate:
            return modified_preds, chamfer_loss, gen_cloud_mod, intermediate_results
        else:
            return modified_preds, chamfer_loss, gen_cloud_mod


# multi element Adam with single loss
# set direction weight to finetune with directional chamfer loss
# includes additional fix for elbows which have thick edges
def chamfer_fine_tune_elbow_fix(
    n_iter,
    step_size,
    preds,
    cloud,
    cat,
    blueprint,
    alpha=1.0,
    visualise=True,
    elbow_fix=True,
    robust=None,
    delta=0.1,
    bidirectional_robust=True,
    return_intermediate=False,
    k=1,
    direction_weight=None,
    loss_func= "chamfer"
):
    # prepare data on gpu and setup optimiser
    cuda = torch.device("cuda")
    preds_copy = copy.deepcopy(preds)
    scaled_original_preds = [scale_preds(pc.tolist(), cat) for pc in preds_copy]

    # temporarily change to socket category for chamfer loss calculations
    if not elbow_fix:
        cat = "socket"
        l = np.array([[p[0] * 0.9] for p in preds])
        preds = list(np.hstack((preds, l)).astype(np.float32))

    preds_t = torch.tensor(preds, requires_grad=True, device=cuda)
    cloud_t = torch.tensor(cloud, device=cuda)
    optimiser = torch.optim.Adam([preds_t], lr=step_size)

    if not elbow_fix:
        a_s_t, a_c_t = torch.clone(preds_t[:, 6]), torch.clone(preds_t[:, 7])

    # check initial loss
    chamfer_loss, gen_cloud = get_chamfer_loss_tensor(
        preds_t, cloud_t, cat, reduce=False, return_cloud=True
    )
    gen_cloud = gen_cloud.detach().cpu().numpy()
    # print("intial loss", chamfer_loss)

    intermediate_results = []

    # iterative refinement with adam
    for i in tqdm(range(n_iter)):
        optimiser.zero_grad()
        if return_intermediate:
            intermediate_results.append(preds_t.clone().detach().cpu().numpy())

        if direction_weight is None:
            if loss_func == "chamfer":
                chamfer_loss = get_chamfer_loss_tensor(
                    preds_t,
                    cloud_t,
                    cat,
                    alpha=alpha,
                    robust=robust,
                    delta=delta,
                    bidirectional_robust=bidirectional_robust,
                )
            elif loss_func == "pair":
                chamfer_loss = get_pair_loss_tensor(preds_t, cloud_t, cat)
            elif loss_func == "emd":
                chamfer_loss = get_emd_loss_tensor(preds_t, cloud_t, cat)
        else:
            chamfer_loss = get_chamfer_loss_directional_tensor(
                preds_t,
                cloud_t,
                cat,
                alpha=alpha,
                robust=robust,
                delta=delta,
                bidirectional_robust=bidirectional_robust,
                k=k,
                direction_weight=direction_weight,
            )

        chamfer_loss.backward()
        optimiser.step()
        if not elbow_fix:
            with torch.no_grad():
                preds_t[:, 6], preds_t[:, 7] = a_s_t, a_c_t

        print(i, "loss", chamfer_loss.detach().cpu().numpy())  # , "preds", preds_t)

    # check final loss
    chamfer_loss, gen_cloud_mod = get_chamfer_loss_tensor(
        preds_t, cloud_t, cat, reduce=False, return_cloud=True
    )
    gen_cloud_mod = gen_cloud_mod.detach().cpu().numpy()

    # print("final loss", chamfer_loss)
    modified_preds = preds_t.detach().cpu().numpy()

    if cat == "socket":
        cat = "elbow"
        modified_preds = modified_preds[:, :-1]

    # additional fix for elbows
    if elbow_fix and cat == "elbow":
        modified_preds, gen_cloud_mod = elbow_correction(
            n_iter, step_size * 2, modified_preds, cloud, blueprint, gen_cloud_mod
        )
        print("error cloud", gen_cloud_mod[0].shape)
    # visualise
    if visualise:
        error_count = 0
        scaled_preds = [scale_preds(p.tolist(), cat) for p in modified_preds]
        visualisers = []
        cloud_visualisers = []

        for i, p in enumerate(scaled_preds):
            try:
                v_orignal, _ = visualize_predictions(
                    [cloud[i].transpose(1, 0).tolist()],
                    cat,
                    [scaled_original_preds[i]],
                    blueprint,
                    visualize=True,
                )
                v_modified, _ = visualize_predictions(
                    [None, None, cloud[i].transpose(1, 0).tolist()],
                    cat,
                    [scaled_preds[i]],
                    blueprint,
                    visualize=True,
                )
                visualisers.append(v_orignal)
                visualisers.append(v_modified)
            except:
                print("error", i)
                error_count += 1
                v_orignal, _ = visualize_predictions(
                    [cloud[i].transpose(1, 0).tolist(), gen_cloud[i].tolist()],
                    cat,
                    [],
                    blueprint,
                    visualize=True,
                )
                v_modified, _ = visualize_predictions(
                    [
                        None,
                        gen_cloud_mod[i].tolist(),
                        cloud[i].transpose(1, 0).tolist(),
                    ],
                    cat,
                    [],
                    blueprint,
                    visualize=True,
                )
                cloud_visualisers.append(v_orignal)
                cloud_visualisers.append(v_modified)

        #         return v_orignal,v_modified, modified_preds
        print("errors ", error_count)
        return cloud_visualisers, visualisers, modified_preds
    else:
        if elbow_fix:
            return modified_preds
        else:
            if return_intermediate:
                return modified_preds, chamfer_loss, gen_cloud_mod, intermediate_results
            else:
                return modified_preds, chamfer_loss, gen_cloud_mod


# fine tune with mahalanobis loss instead of chamfer loss
def mahalanobis_fine_tune(
    n_iter,
    step_size,
    preds,
    cloud,
    means,
    covariances,
    cat,
    blueprint,
    alpha=1,
    visualise=True,
    elbow_fix=True,
    robust=None,
    delta=0.1,
    chamfer=0,
    weights=None,
):
    # prepare data on gpu and setup optimiser
    cuda = torch.device("cuda")
    preds_copy = copy.deepcopy(preds)
    scaled_original_preds = [scale_preds(pc.tolist(), cat) for pc in preds_copy]

    # temporarily change to socket category for chamfer loss calculations
    if not elbow_fix:
        cat = "socket"
        l = np.array([[p[0] * 0.9] for p in preds])
        preds = list(np.hstack((preds, l)).astype(np.float32))

    preds_t = torch.tensor(preds, requires_grad=True, device=cuda)
    cloud_t = torch.tensor(cloud, device=cuda)
    optimiser = torch.optim.Adam([preds_t], lr=step_size)

    if not elbow_fix:
        a_s_t, a_c_t = torch.clone(preds_t[:, 6]), torch.clone(preds_t[:, 7])

    # check initial loss
    mahalanobis_loss, gen_cloud = get_mahalanobis_loss_tensor(
        preds_t, means, covariances, cat, return_cloud=True
    )
    gen_cloud = gen_cloud.detach().cpu().numpy()
    # print("intial loss", chamfer_loss)

    intermediate_results = []

    # iterative refinement with adam
    for i in tqdm(range(n_iter)):
        optimiser.zero_grad()
        mahalanobis_loss = get_mahalanobis_loss_tensor(
            preds_t,
            means,
            covariances,
            cat,
            robust=robust,
            delta=delta,
            chamfer=chamfer,
            alpha=alpha,
            src_pcd_tensor=cloud_t,
            weights=weights,
        )
        mahalanobis_loss.backward()
        optimiser.step()
        if not elbow_fix:
            with torch.no_grad():
                preds_t[:, 6], preds_t[:, 7] = a_s_t, a_c_t

        print(i, "loss", mahalanobis_loss.detach().cpu().numpy())  # , "preds", preds_t)

    # check final loss
    mahalanobis_loss, gen_cloud_mod = get_mahalanobis_loss_tensor(
        preds_t, means, covariances, cat, return_cloud=True
    )
    gen_cloud_mod = gen_cloud_mod.detach().cpu().numpy()

    # print("final loss", chamfer_loss)
    modified_preds = preds_t.detach().cpu().numpy()

    if cat == "socket":
        cat = "elbow"
        modified_preds = modified_preds[:, :-1]

    # additional fix for elbows
    if elbow_fix and cat == "elbow":
        modified_preds, gen_cloud_mod = elbow_correction(
            n_iter, step_size * 2, modified_preds, cloud, blueprint, gen_cloud_mod
        )
        print("error cloud", gen_cloud_mod[0].shape)
    # visualise
    if visualise:
        error_count = 0
        scaled_preds = [scale_preds(p.tolist(), cat) for p in modified_preds]
        visualisers = []
        cloud_visualisers = []

        for i, p in enumerate(scaled_preds):
            try:
                v_orignal, _ = visualize_predictions(
                    [cloud[i].transpose(1, 0).tolist()],
                    cat,
                    [scaled_original_preds[i]],
                    blueprint,
                    visualize=True,
                )
                v_modified, _ = visualize_predictions(
                    [None, None, cloud[i].transpose(1, 0).tolist()],
                    cat,
                    [scaled_preds[i]],
                    blueprint,
                    visualize=True,
                )
                visualisers.append(v_orignal)
                visualisers.append(v_modified)
            except:
                print("error", i)
                error_count += 1
                v_orignal, _ = visualize_predictions(
                    [cloud[i].transpose(1, 0).tolist(), gen_cloud[i].tolist()],
                    cat,
                    [],
                    blueprint,
                    visualize=True,
                )
                v_modified, _ = visualize_predictions(
                    [
                        None,
                        gen_cloud_mod[i].tolist(),
                        cloud[i].transpose(1, 0).tolist(),
                    ],
                    cat,
                    [],
                    blueprint,
                    visualize=True,
                )
                cloud_visualisers.append(v_orignal)
                cloud_visualisers.append(v_modified)

        #         return v_orignal,v_modified, modified_preds
        print("errors ", error_count)
        return cloud_visualisers, visualisers, modified_preds
    else:
        if elbow_fix:
            return modified_preds
        else:
            if return_intermediate:
                return modified_preds, chamfer_loss, gen_cloud_mod, intermediate_results
            else:
                return modified_preds, chamfer_loss, gen_cloud_mod


# single thread, adam optimiser
def chamfer_fine_tune_single(n_iter, step_size, preds, cloud, cat, blueprint):
    print(cloud.shape)
    cuda = torch.device("cuda")
    # cloud = cloud.transpose(1,0)
    preds_copy = copy.deepcopy(preds)
    scaled_original_preds = scale_preds(preds_copy.tolist(), cat)
    preds_t = torch.tensor([preds], requires_grad=True, device=cuda)
    cloud_t = torch.tensor([cloud], device=cuda)

    optimiser = torch.optim.Adam([preds_t], lr=step_size)

    for i in tqdm(range(n_iter)):
        t1 = time.perf_counter()

        optimiser.zero_grad()
        chamfer_loss = get_chamfer_loss_tensor(preds_t, cloud_t, cat)
        t2 = time.perf_counter()
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        chamfer_loss.backward()
        optimiser.step()

        t3 = time.perf_counter()
        # print("chamf", t2-t1, "grad", t3-t2)

        print(i, "loss", chamfer_loss.item(), "preds", preds_t.detach().cpu().numpy())

    # visualise
    modified_preds = preds_t.detach().cpu().numpy()[0].tolist()
    scaled_preds = scale_preds(modified_preds, cat)
    v_orignal, _ = visualize_predictions(
        [cloud.transpose(1, 0).tolist()],
        cat,
        [scaled_original_preds],
        blueprint,
        visualize=True,
    )
    v_modified, _ = visualize_predictions(
        [cloud.transpose(1, 0).tolist()], cat, [scaled_preds], blueprint, visualize=True
    )

    return [v_orignal, v_modified], chamfer_loss.item()
