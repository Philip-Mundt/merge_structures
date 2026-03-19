import numpy as np
import MDAnalysis as mda
from tqdm.notebook import tqdm
import math

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
    
def place_clusters(target_univ, clusters, z_offset):
    """
    Places clusters known to fit into the universe
    """
    if not clusters:
        return target_univ

    # Collect existing atoms and translated new atoms
    to_combine = [target_univ.atoms]
    new_placed_clusters = []

    for cluster_dict in clusters:
        cluster_dict_copy = cluster_dict.copy()
        cluster_ag = cluster_dict_copy["cluster"]
        cluster_ag.translate([0, 0, z_offset])
        to_combine.append(cluster_ag)
        # update cluster positions also for placed clusters
        current_center = cluster_dict["center"]
        cluster_dict_copy["center"] = [current_center[0], current_center[1], (current_center[2] + z_offset) % target_univ.dimensions[2]]
        new_placed_clusters.append(cluster_dict_copy)
    # combine all clusters at once
    new_univ = mda.Merge(*to_combine)
    # restore original box
    new_univ.dimensions = target_univ.dimensions
    # translate overhanging atoms over periodic boundaries 
    new_univ.atoms.wrap(compound="segments")
    return new_univ, new_placed_clusters

def cylinders_overlap(c1, c2, box):
    """
    Checks if two cylinders overlap in a periodic box.
    Returns True if they overlap and False otherwise.
    """
    xdim, ydim, zdim = box[:3]

    # Distances between centers
    dx = c1["center"][0] - c2["center"][0]
    dy = c1["center"][1] - c2["center"][1]
    dz = c1["center"][2] - c2["center"][2]

# Shortest distance across periodic boundaries
    dx -= xdim * np.round(dx/xdim)
    dy -= ydim * np.round(dy/ydim)
    dz -= zdim * np.round(dz/zdim)
    # same as:
    # dx = dx % xdim
    # if dx > xdim/2:
    #     dx = xdim - dx
    # ...

    dxy = np.sqrt(dx*dx + dy*dy)
    
    # check xy overlap (+ 10 Å buffer for bead sizes)
    # if not: no intersection possible
    if dxy >= (c1["radius"] + c2["radius"] + 20):
        return False

    # check Z overlap (+ 10 Å buffer for bead sizes)
    if abs(dz) >= (c1["height"]/2 + c2["height"]/2 + 20):
        return False

    return True

def evaluate_structure_fitting(target_clusters, clusters, n_atoms, box,
                               coarse_step=150, fine_step=20):

    zdim = box[2]

    def evaluate_z(z_offset, return_clusters=False):

        fitted_atoms = 0
        fitting = []
        orphan = []

        for cluster in clusters:
            center = cluster["center"].copy()
            # add z offset
            center[2] = (center[2] + z_offset) % zdim

            test_cyl = {
                "center": center,
                "radius": cluster["radius"],
                "height": cluster["height"]
            }

            collision = False

            for placed in target_clusters:
                if cylinders_overlap(test_cyl, placed, box):
                    collision = True
                    break

            if not collision:
                fitted_atoms += len(cluster["cluster"].atoms)
                if return_clusters:
                    fitting.append(cluster)
            else:
                if return_clusters:
                    orphan.append(cluster)

        score = (fitted_atoms / n_atoms) / len(clusters)

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

        score, fitting, orphan = evaluate_z(z, return_clusters=True)

        if score > best_params["fitness_score"]:
            best_params = {
                "fitness_score": score,
                "z_offset": z,
                "fitting_clusters": fitting,
                "orphan_clusters": orphan
            }

    return best_params

def merge_structures(proteins_df):

    # calculate box dimensions for the new structure
    x_dim, y_dim, z_dim = calculate_box_from_condensate(proteins_df)
    # ***************************************************
    # Test
    # x_dim, y_dim, z_dim = box = x_dim, y_dim, 2000
    # ***************************************************
    print(f"New Structure file initialized with box dimensions [{x_dim}, {y_dim}, {z_dim}].")

    placing_queue = proteins_df.copy()

    placing_queue["n_clusters"] = placing_queue["subaggregates"].apply(len)
    placing_queue = placing_queue.sort_values("n_clusters")

    pbar = tqdm(total=len(placing_queue), desc="placing remaining structures")

    first = placing_queue.index[0]
    merged = placing_queue.loc[first,"universe"].copy()
    first_clusters = placing_queue.loc[first, "subaggregates"]
    first_box = placing_queue.loc[first, "box_dimensions"]

    pbar.write(f"Placed structure {first} ({placing_queue.loc[first, "n_clusters"]} clusters).")
    pbar.update(1)

    placing_queue.drop(first, inplace=True)

    # changing box dimensions to calculated & centering structures in the new box
    merged.dimensions = [x_dim, y_dim, z_dim, 90, 90, 90]
    shift = (np.array([x_dim, y_dim, z_dim]) / 2) - (first_box[:3] / 2)
    merged.atoms.translate(shift)
    for cluster_data in first_clusters:
        cluster_data["center"] += shift 
        # not necessary but maybe will come in handy later (?)
        cluster_data["cluster"].atoms.translate(shift)
    for row in placing_queue.itertuples():
        u = row.universe
        original_box = u.dimensions.copy()
        u.dimensions = [x_dim, y_dim, z_dim, 90, 90, 90]
        shift = (np.array([x_dim, y_dim, z_dim]) / 2) - (original_box[:3] / 2)
        u.atoms.translate(shift)
        for cluster_data in row.subaggregates:
            cluster_data["center"] += shift
        u.atoms.wrap(compound="segments")


    # list so collect all clusters placed and those not yet placed
    placed_clusters = []
    placed_clusters.extend(first_clusters)
    orphan_clusters = []

    while not placing_queue.empty:

        placing_queue["fitness_results"] = placing_queue.apply(
            lambda row: evaluate_structure_fitting(
                placed_clusters,
                row["subaggregates"],
                row["n_atoms"],
                [x_dim,y_dim,z_dim]
            ),
            axis=1
        )

        placing_queue["score"] = placing_queue["fitness_results"].apply(
            lambda x: x["fitness_score"]
        )

        placing_queue = placing_queue.sort_values("score", ascending=False)

        best = placing_queue.index[0]
        params = placing_queue.loc[best,"fitness_results"]

        merged, new_placed_clusters = place_clusters(merged, params["fitting_clusters"], params["z_offset"])
        placed_clusters.extend(new_placed_clusters)
        new_orphan_clusters = params["orphan_clusters"]
        orphan_clusters.extend(new_orphan_clusters)
        pbar.write(f"Placed structure {best} ({len(new_placed_clusters)}/{len(new_orphan_clusters) + len(new_placed_clusters)})")

        placing_queue.drop(best, inplace=True)

        pbar.update(1)

    pbar.close()

    merged.atoms.wrap(compound="segments")

    return merged