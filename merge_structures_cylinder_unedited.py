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

def compute_cluster_cylinder(cluster):
    pos = cluster.positions
    center = cluster.center_of_geometry()

    xy = pos[:, :2] - center[:2]
    radius = np.sqrt((xy**2).sum(axis=1)).max()

    zmin = pos[:,2].min()
    zmax = pos[:,2].max()
    center[2] = (zmin+zmax)/2

    return {
        "cluster": cluster,
        "com": center,
        "radius": radius,
        "height": zmax - zmin
    }


def cylinders_overlap(c1, c2, box):
    xdim, ydim, zdim = box[:3]

    dx = c1["x"] - c2["x"]
    dy = c1["y"] - c2["y"]

    dx -= xdim * np.round(dx/xdim)
    dy -= ydim * np.round(dy/ydim)

    dxy = np.sqrt(dx*dx + dy*dy)

    if dxy >= (c1["radius"] + c2["radius"]):
        return False

    if c1["z_high"] < c2["z_low"] or c1["z_low"] > c2["z_high"]:
        return False

    return True

#     # Distances between centers
#     dx = c1["center"][0] - c2["center"][0]
#     dy = c1["center"][1] - c2["center"][1]
#     dz = c1["center"][2] - c2["center"][2]

# # Shortest distance across periodic boundaries
#     dx -= xdim * np.round(dx/xdim)
#     dy -= ydim * np.round(dy/ydim)
#     dz -= zdim * np.round(dz/zdim)
#     # same as:
#     # dx = dx % xdim
#     # if dx > xdim/2:
#     #     dx = xdim - dx
#     # ...

#     dxy = np.sqrt(dx*dx + dy*dy)
    
#     # check xy overlap (+ 10 Å buffer for bead sizes)
#     # if not: no intersection possible
#     if dxy >= (c1["radius"] + c2["radius"] + 20):
#         return False

#     # check Z overlap (+ 10 Å buffer for bead sizes)
#     if abs(dz) >= (c1["height"]/2 + c2["height"]/2 + 20):
#         return False

#     return True



def evaluate_structure_fitting(target_cylinders, clusters, n_atoms, box,
                               coarse_step=150, fine_step=20):

    zdim = box[2]

    cluster_cyl = [compute_cluster_cylinder(c) for c in clusters]

    def evaluate_z(z_offset, return_clusters=False):

        fitted_atoms = 0
        fitting = []
        orphan = []

        for cyl in cluster_cyl:

            com = cyl["com"]

            z = (com[2] + z_offset) % zdim

            test_cyl = {
                "x": com[0],
                "y": com[1],
                "radius": cyl["radius"],
                "z_low": z - cyl["height"]/2,
                "z_high": z + cyl["height"]/2
            }

            collision = False

            for placed in target_cylinders:
                if cylinders_overlap(test_cyl, placed, box):
                    collision = True
                    break

            if not collision:
                fitted_atoms += len(cyl["cluster"].atoms)
                if return_clusters:
                    fitting.append(cyl["cluster"])
            else:
                if return_clusters:
                    orphan.append(cyl["cluster"])

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

        score, fit, orphan = evaluate_z(z, True)

        if score > best_params["fitness_score"]:
            best_params = {
                "fitness_score": score,
                "z_offset": z,
                "fitting_clusters": fit,
                "orphan_clusters": orphan
            }

    return best_params


def place_clusters(target_univ, clusters, z_offset):
    """
    Places clusters known to fit into the universe
    """
    if not clusters:
        return target_univ
    
    # Collect existing atoms and translated new atoms
    to_combine = [target_univ.atoms]
    for cluster in clusters:
        cluster_copy = cluster.copy()
        cluster_copy.translate([0, 0, z_offset])
        to_combine.append(cluster_copy)
    # combine all clusters at once
    new_univ = mda.Merge(*to_combine)
    # restore original box
    new_univ.dimensions = target_univ.dimensions
    # translate overhanging atoms over periodic boundaries 
    new_univ.atoms.wrap(compound="segments")
    return new_univ

def merge_structures(proteins_df):

    placing_queue = proteins_df.copy()

    placing_queue["n_clusters"] = placing_queue["subaggregates"].apply(len)
    placing_queue = placing_queue.sort_values("n_clusters")

    first = placing_queue.index[0]
    merged = placing_queue.loc[first,"universe"].copy()

    placing_queue.drop(first, inplace=True)

    x_dim, y_dim, z_dim = calculate_box_from_condensate(proteins_df)
    merged.dimensions = [x_dim,y_dim,z_dim,90,90,90]

    merged.atoms.translate([
        x_dim/2, y_dim/2, z_dim/2
    ] - merged.atoms.center_of_geometry())

    placed_cylinders = []

    for seg in merged.segments:
        cyl = compute_cluster_cylinder(seg.atoms)
        placed_cylinders.append({
            "x": cyl["com"][0],
            "y": cyl["com"][1],
            "radius": cyl["radius"],
            "z_low": cyl["com"][2] - cyl["height"]/2,
            "z_high": cyl["com"][2] + cyl["height"]/2
        })

    pbar = tqdm(total=len(placing_queue), desc="placing structures")

    while not placing_queue.empty:

        placing_queue["fitness_results"] = placing_queue.apply(
            lambda row: evaluate_structure_fitting(
                placed_cylinders,
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

        merged = place_clusters(merged,
                                params["fitting_clusters"],
                                params["z_offset"])

        for cluster in params["fitting_clusters"]:

            cyl = compute_cluster_cylinder(cluster)

            placed_cylinders.append({
                "x": cyl["com"][0],
                "y": cyl["com"][1],
                "radius": cyl["radius"],
                "z_low": cyl["com"][2] - cyl["height"]/2,
                "z_high": cyl["com"][2] + cyl["height"]/2
            })

        placing_queue.drop(best, inplace=True)

        pbar.update(1)

    pbar.close()

    merged.atoms.wrap(compound="segments")

    return merged
