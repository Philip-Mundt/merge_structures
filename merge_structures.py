# imports
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
import math
from tqdm.notebook import tqdm


def calculate_box_from_condensate(proteins_df, sigmas=4):
    """
    Calculates box dimensions based on:
    1. XY size of the box cannot be smaller then any of the original xy box sizes
    2. Avoiding self-interaction (XY > 2*Rg of the largest protein)
    3. Ability of condensates to span the entire box in XY direction (XY << V(all particles)/sigma)
    4. Rough box shape (8*XY <= Z <= 10*XY)
    5. Overall protein concentration (V(box) ~ concentration)

    :param universe_dict: dictionary with {prot_name: universe, subaggregates, chain_size}
    :param sigmas: average size of beads [Ångström]
    """
    # Constraint 1:
    # the largest x or y size of an input box is set as minimum for the output box 
    # to make sure no big aggregate is crammed into a box that is too thin
    min_xy = np.array(proteins_df["box_dimensions"].tolist())[:, :2].max()    
    
    # Constraint 2: 
    min_xy = max(min_xy, (2 * proteins_df["rg_max"].max()))

    # Constraint 3: 
    # volume is approximated by just adding the volumes of boxes surrounding a bead and adding them together plus a bit of wiggle room
    max_xy = math.sqrt(sigmas**2 * proteins_df["n_atoms"].min()) * 1.5
    max_xy = max(max_xy, min_xy)

    # Constraint 4: 
    min_z = min_xy * 8
    max_z = max_xy * 10

    # Constraint 5:
    # calculate mean concentration of inputs
    volumes = [dim[0]*dim[1]*dim[2] for dim in proteins_df["box_dimensions"]]
    prot_concs = proteins_df["n_proteins"].to_numpy() / np.array(volumes)
    prot_conc = prot_concs.mean()
    # calculate 
    n_prot_total = proteins_df["n_proteins"].sum()
    target_volume = n_prot_total / prot_conc
    
    # 
    if target_volume < min_xy**2 * min_z:
        return min_xy, min_xy, min_z
    elif target_volume > max_xy**2 * max_z:
        return max_xy, max_xy, max_z
    else:
        xy = np.cbrt(target_volume/9)
        z = xy * 9
        return xy, xy, z

def cylinders_overlap(c1, c2, box, sigma=6):
    """
    Checks if cluster cylinders overlap (plus some buffer for the bead size).

    :param c1: protein cluster to check
    :type c1: AtomGroup
    :param c2: protein cluster to check against
    :type c2: AtomGroup
    :param box: Box in which both clusters are/should be placed
    :type box: List/NumPy array
    :returns: True if clusters overlap, else False
    :rtype: bool
    """
    xdim, ydim, zdim = box[:3]

    # distances between centers
    dx = c1["center"][0] - c2["center"][0]
    dy = c1["center"][1] - c2["center"][1]
    dz = c1["center"][2] - c2["center"][2]
    # distance to closest image
    dx -= xdim * np.round(dx/xdim)
    dy -= ydim * np.round(dy/ydim)
    dz -= zdim * np.round(dz/zdim)

    dxy = np.sqrt(dx*dx + dy*dy)
    # check xy overlap (+ 2*sigma Å buffer for bead sizes)
    # if not: no intersection possible
    if dxy >= (c1["radius"] + c2["radius"] + sigma*2):
        return False

    # check Z overlap (+ 10 Å buffer for bead sizes)
    if abs(dz) >= (c1["height"]/2 + c2["height"]/2 + sigma*2):
        return False
    
    return True

def evaluate_structure_fitting(target_cylinders, cluster_cyl, n_atoms, target_box,
                               coarse_step=150, fine_step=20):

    zdim = target_box[2]

    cluster_cyl

    def evaluate_z(z_offset, return_clusters=False):

        fitted_atoms = 0
        fitting = []
        orphan = []

        for cyl in cluster_cyl:

            center = cyl["center"]

            z = (center[2] + z_offset) % zdim

            test_cyl = {
                "center": [center[0], center[1], z],
                "radius": cyl["radius"],
                "height": cyl["height"]
            }

            collision = False

            for placed in target_cylinders:
                if cylinders_overlap(test_cyl, placed, target_box):
                    collision = True
                    break

            if not collision:
                fitted_atoms += len(cyl["cluster"].atoms)
                if return_clusters:
                    fitting.append(cyl)
            else:
                if return_clusters:
                    orphan.append(cyl)

        score = (fitted_atoms / n_atoms) / len(cluster_cyl)

        if return_clusters:
            return score, fitting, orphan

        return score


    best_z = 0
    best_score = -1

    for z in range(0, int(zdim), coarse_step):
        score = evaluate_z(z)

        if score > best_score:
            best_score = score
            best_z = z

    best_params = {
        "fitness_score": -1,
        "z_offset": 0,
        "fitting_clusters": [],
        "orphan_clusters": []
    }

    for z in range(best_z - coarse_step, best_z + coarse_step, fine_step):

        score, fit, orphan = evaluate_z(z, True)

        if score > best_params["fitness_score"]:
            best_params = {
                "fitness_score": score,
                "z_offset": z,
                "fitting_clusters": fit,
                "orphan_clusters": orphan
            }

    return best_params

def place_clusters(target_univ, clusters, offset, box, independent=False):
    """
    Places clusters known to fit into the universe
    """
    if not clusters:
        return target_univ
    
    # Collect existing atoms and translated new atoms
    to_combine = [target_univ.atoms]
    shift = offset
    for i in range(len(clusters)):
        cluster_copy = clusters[i].copy()
        if offset is not None:
            if independent:
                shift = offset[i]
            cluster_copy.translate(shift)
        to_combine.append(cluster_copy)
    # combine all clusters at once
    new_univ = mda.Merge(*to_combine)
    # restore original box
    new_univ.dimensions = target_univ.dimensions
    # translate overhanging atoms over periodic boundaries 
    new_univ.atoms.wrap(compound="atoms", box=[box[0], box[1], box[2], 90, 90, 90])
    return new_univ

def merge_structures(proteins_df):

    # calculate box dimensions for the new structure
    x_dim, y_dim, z_dim = calculate_box_from_condensate(proteins_df)
    # ***************************************************
    # Test
    # x_dim, y_dim, z_dim = x_dim, y_dim, 1800
    # ***************************************************
    box = np.array([x_dim, y_dim, z_dim])
    print(f"New Structure file initialized with box dimensions [{x_dim}, {y_dim}, {z_dim}].")


    placing_queue = proteins_df.copy()
    # start with the structure with the least clusters
    placing_queue["n_clusters"] = placing_queue["subaggregates"].apply(len)
    placing_queue = placing_queue.sort_values("n_clusters")

    pbar = tqdm(total=len(placing_queue), desc="Placing structures")

    first = placing_queue.index[0]
    merged = placing_queue.loc[first,"universe"].copy()
    original_box = placing_queue.loc[first, "box_dimensions"]
    original_cylinders = placing_queue.loc[first, "subaggregates"]

    pbar.write(f"Placed structure {first} ({placing_queue.loc[first, "n_clusters"]} clusters).")
    pbar.update(1)


    # changing box dimensions to calculated & centering structures in the new box
    merged.dimensions = [x_dim, y_dim, z_dim, 90, 90, 90]
    shift = (np.array([x_dim, y_dim, z_dim]) / 2) - (original_box[:3] / 2)
    merged.atoms.translate(shift)
    merged.atoms.wrap(compound="atoms")
    # adjust cylinder & cluster atoms positions
    for cyl in original_cylinders:
        center = cyl["center"]
        new_center = (np.array(center) + shift) % box
        cyl["center"] = new_center
        cyl["cluster"].translate(shift)
        cyl["cluster"].wrap(compound="atoms", box=[box[0], box[1], box[2], 90, 90, 90])
    placed_cylinders = original_cylinders.copy()


    for row in placing_queue.itertuples():
        u = row.universe
        orig_box = row.box_dimensions
        u.dimensions = [x_dim, y_dim, z_dim, 90, 90, 90]
        shift = (np.array([x_dim, y_dim, z_dim]) / 2) - (orig_box[:3] / 2)
        u.atoms.translate(shift)
        u.atoms.wrap(compound="atoms")
        for cyl in row.subaggregates:
            center = cyl["center"]
            new_center = (np.array(center) + shift) % box
            cyl["center"] = new_center

    placing_queue.drop(first, inplace=True)

    # list so collect all clusters not yet placed
    orphan_cylinders = []

    while not placing_queue.empty:
        # check how well and where the remaining structures fit best into the new box
        placing_queue["fitness_results"] = placing_queue.apply(
            lambda row: evaluate_structure_fitting(
                placed_cylinders,
                row["subaggregates"],
                row["n_atoms"],
                [x_dim,y_dim,z_dim]
            ),
            axis=1
        )

        # sort the queue to continue placing with the best fitting structure
        placing_queue["score"] = placing_queue["fitness_results"].apply(lambda x: x["fitness_score"])
        placing_queue = placing_queue.sort_values("score", ascending=False)
        best = placing_queue.index[0]
        fit_dict = placing_queue.loc[best,"fitness_results"]

        # add the fitting clusters of the best fitting structure to the new structure
        merged = place_clusters(merged, [cyl["cluster"] for cyl in fit_dict["fitting_clusters"]], [0, 0, fit_dict["z_offset"]], box=box)
        # update cylinders for the placed aggregates in their new place
        for cyl in fit_dict["fitting_clusters"]:
            center = cyl["center"]
            new_center = (np.array(center) + np.array([0, 0, fit_dict["z_offset"]])) % box
            cyl["center"] = new_center 
            placed_cylinders.append(cyl)
        
        for orphan in fit_dict["orphan_clusters"]:
            # translate orphans too so they stay close to their original position in the structure
            orphan["cluster"].translate([0, 0, fit_dict["z_offset"]])
            orphan["cluster"].wrap(compound="atoms", box=[box[0], box[1], box[2], 90, 90, 90])
            center = orphan["center"]
            new_center = (np.array(center) + np.array([0, 0, fit_dict["z_offset"]])) % box
            orphan["center"] = new_center 
            orphan_cylinders.append(orphan)

        
        pbar.write(f"Placed structure {best} ({len(fit_dict["fitting_clusters"])}/{len(fit_dict["fitting_clusters"]) + len(fit_dict["orphan_clusters"])})")

        placing_queue.drop(best, inplace=True)

        pbar.update(1)

    pbar.close()


    # handle clusters that could not yet be placed
    # --- ORPHAN HANDLING BLOCK ---
    n_orphans = len(orphan_cylinders)
    fitting_orphans, misfit_orphans = find_orphan_placement(orphan_cylinders, placed_cylinders, box)
    merged = place_clusters(merged, [cyl["cluster"] for cyl in fitting_orphans], box=box, offset=None)
    placed_cylinders.extend(fitting_orphans)
    print(f"{len(fitting_orphans)}/{n_orphans} clusters were placed.")
    # if misfit_orphans:
    #     print(f"Start more thorough orphan handling search for remaining {len(misfit_orphans)} orphans.")
    #     fitting_misfits, misfit_misfits = find_orphan_placement(misfit_orphans, placed_cylinders, box, thorough=True)
    #     merged = place_clusters(merged, [cyl["cluster"] for cyl in fitting_misfits], box=box, offset=None)
    #     print(f"{len(misfit_misfits)} clusters could not be placed.")


    merged.atoms.wrap(compound="segments")

    return merged

def find_orphan_placement(orphan_cylinders, placed_cylinders, box, thorough=False):    
    pbar = tqdm(total=len(orphan_cylinders), desc=f"Fitting {"misfit " if thorough else ""}orphans")
    # Sort orphans by cylinder size (largest first)
    orphan_cylinders.sort(key=lambda x: x["radius"] * x["height"], reverse=True)

    fitting_orphans = []
    misfit_orphans = []
    placed_cylinders_copy = placed_cylinders.copy()
    step_sizes = [50, 20] if thorough else [50]
    for orphan_cyl in orphan_cylinders:
        success = False
        for step_size in step_sizes:
            for z_offset in range(0, int(box[2]), step_size):
                for y_offset in range(0, int(box[1]), step_size):
                    for x_offset in range(0, int(box[0]), step_size):
                        fits = test_orphan_offset(
                            cylinder_dict=orphan_cyl, 
                            target_cylinders=placed_cylinders_copy, 
                            box=box[:3], 
                            offset=[x_offset, y_offset, z_offset],
                            thorough=thorough
                            )
                        if fits:
                            print(f"Found spot for orphan at {x_offset}, {y_offset}, {z_offset}")
                            # Create a copy of the dict to avoid modifying the one in orphan_cylinders
                            placed_cyl = orphan_cyl.copy()
                            # Apply the offset to the center for future collision checks
                            new_center = (np.array(orphan_cyl["center"]) + np.array([x_offset, y_offset, z_offset])) % box
                            placed_cyl["center"] = new_center
                            placed_cluster = placed_cyl["cluster"].copy()
                            placed_cluster.translate([x_offset, y_offset, z_offset])
                            placed_cluster.wrap(compound="atoms", box=[box[0], box[1], box[2], 90, 90, 90])
                            placed_cyl["cluster"] = placed_cluster
                            # collect orphans for later placement
                            fitting_orphans.append(placed_cyl)
                            # save the cylinder 
                            placed_cylinders_copy.append(placed_cyl)
                            success = True
                            break
                    if success:
                        break
                if success:
                    break
            if success:
                break
        if not success:
            print("No place found for orphan.")
            misfit_orphans.append(orphan_cyl)
        pbar.update(1)  
    pbar.close()
 
    return fitting_orphans, misfit_orphans

def test_orphan_offset(cylinder_dict, target_cylinders, box, offset, sigma=6, thorough=False):
    """
    Test if a cylinder with a certain offset fits in the current structure.

    """
    center = np.array(cylinder_dict["center"])
    center = (center + np.array(offset)) % np.array(box)
    test_cyl = {
        "center": center,
        "radius": cylinder_dict["radius"],
        "height": cylinder_dict["height"]
    }
    if thorough:
        test_cluster = cylinder_dict["cluster"].copy()
        test_cluster.translate(offset)
        test_cluster.wrap(compound="atoms", box=[box[0], box[1], box[2], 90, 90, 90])
        test_cyl["cluster"] = test_cluster
    fits = True

    for placed in target_cylinders:
        if cylinders_overlap(test_cyl, placed, box):
            if thorough:
                # for overlapping cylinders look if the structures actually overlap
                pairs = capped_distance(
                    reference=placed["cluster"], 
                    configuration=test_cyl["cluster"], 
                    max_cutoff=sigma*2, 
                    box=[box[0], box[1], box[2], 90, 90, 90]
                    )
                if len(pairs[0]) == 0:
                    # if there is no actual overlap of the structures go on to the next
                    continue
            fits = False
            break

    return fits
    