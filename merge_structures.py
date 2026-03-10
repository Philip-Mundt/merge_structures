import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import math
from tqdm.notebook import tqdm

def assign_segment_ids(universe, protein_name, existing_segids={}):
    """
    Assigns unique 4-character segment IDs to protein chains.
    
    Format: [Protein Type][Chain Number] (e.g., H201, L101)
    Handles cases where chain IDs repeat in input PDBs.
    
    Args:
        universe: MDAnalysis Universe object
        protein_name: Protein identifier (e.g., "hpl2", "lin13")
        existing_segids: External counter dict to maintain uniqueness across files -> {protein: (segid, nmb)}
    """
    up_protein_name = protein_name.upper()
    # check if protein is known
    if protein_name in existing_segids:
        prefix = existing_segids[protein_name][0]
    else:
        # create a 2 digit short (first character + first number)
        first_letter = up_protein_name[0]
        if any(c.isdigit() for c in protein_name):
            first_digit = next(c for c in protein_name if c.isdigit())
            prefix = first_letter + first_digit
            if prefix in existing_segids:
                prefix = up_protein_name[0:2]
        else: 
            prefix = up_protein_name[0:2]
        if prefix in existing_segids:
                prefix = up_protein_name[0] + "X"
        if prefix in existing_segids:
            while True:
                prefix = input(f"Input a two character short name for the segment IDs of {protein_name}. These names are already taken:")
                print([x[0] for x in existing_segids.values()])
                if len(prefix) == 2:
                    break
                else:
                    print("INvalid input, try again")

    a = universe.atoms

    chain_count = existing_segids.get(protein_name, [0, 0])[1] + 1
    # Track the start position if a new chain
    chain_start = 0

    for i in range(1, len(a)):
        last = a[i-1]
        curr = a[i]
        
        if last.chainID != curr.chainID:
            # Slice based on positions in the current AtomGroup
            current_seg = a[chain_start:i]
            
            seg_id = f"{prefix}{chain_count:02d}"
            # create new segment
            new_seg = universe.add_Segment(segid=seg_id)

            # assign residues to this segment
            current_seg.residues.segments = new_seg

            # Update for next chain
            chain_start = i
            chain_count += 1

    # Get the very last protein chain that the loop finishes on
    last_seg = a[chain_start:]
    seg_id = f"{prefix}{chain_count:02d}"
    new_seg = universe.add_Segment(segid=seg_id)
    last_seg.residues.segments = new_seg
    existing_segids[protein_name] = (prefix, chain_count)
    
    # rebuild universe to clean empty segments
    universe = mda.Merge(a)

    return universe

def identify_subaggregates(universe, cutoff=15.0):
    """
    Clusters proteins if any bead of one is within 'cutoff' of any bead of another.
    """
    # 1. Collect all non-empty protein groups
    prots = [s.atoms for s in universe.segments if len(s.atoms) > 0]
    n_prots = len(prots)
    if n_prots == 0: return []

    # 2. Build Adjacency Matrix
    # Start with an identity matrix (every protein is connected to itself)
    adj = np.eye(n_prots, dtype=bool)

    # 3. Double-loop to find contacts between protein pairs
    for i in range(n_prots):
        for j in range(i + 1, n_prots):
            # capped_distance returns pairs of indices (dist < cutoff)
            # if the returned array is not empty, the proteins are touching
            pairs = capped_distance(prots[i].positions, 
                                    prots[j].positions, 
                                    max_cutoff=cutoff, 
                                    box=universe.dimensions)
            
            if len(pairs[0]) > 0:
                adj[i, j] = adj[j, i] = True

    # 4. Find connected components
    graph = csr_matrix(adj)
    n_clusters, labels = connected_components(graph)

    # 5. Group into AtomGroups
    subaggregates = []
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        # Summing AtomGroups creates a single combined AtomGroup
        cluster_ag = sum(prots[idx] for idx in cluster_indices)
        subaggregates.append(cluster_ag)

    return subaggregates

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

def evaluate_structure_fitting(target_univ, clusters, n_atoms, pbar, coarse_step=150, fine_step=20):
    """
    Finds the best Z-offset for a structure using a two-pass search:
    1. Coarse scan: Quickly find the most promising region of the box.
    2. Fine tuning: Zoom in on that region for precision placement.
    """
    # save the positions of atoms already in the box to speed up checks
    target_pos = target_univ.atoms.positions
    box = target_univ.dimensions
    z_box = int(box[2])
    n_clusters = len(clusters)

    # helper function to calculate fitness scores for sorting structures
    def get_fitness_for_z(test_z, return_clusters=False):
        fitted_atoms = 0
        fitting_list = []
        orphan_list = []
        
        for cluster in clusters:
            # keep cluster within periodic boundaries for checking collisions
            trial_pos = (cluster.positions + [0, 0, test_z])
            trial_pos[:, 2] %= z_box 
            
            # create a (periodic) z slice of the target structure around the cluster to increase performance of collision checks
            z_min_trial = trial_pos[:, 2].min() - 15.0
            z_max_trial = trial_pos[:, 2].max() + 15.0
            # if z slice wrapped around periodic boundaries
            if z_max_trial > z_box or z_min_trial < 0:
                # check top & bottom of the box
                mask = (target_pos[:, 2] <= z_max_trial % z_box) | (target_pos[:, 2] >= z_min_trial % z_box)
            else:
                mask = (target_pos[:, 2] >= z_min_trial) & (target_pos[:, 2] <= z_max_trial)
            
            local_target_pos = target_pos[mask]
            
            # collision check
            if len(local_target_pos) == 0 or len(capped_distance(local_target_pos, trial_pos, max_cutoff=15.0, box=box)[0]) == 0:
                fitted_atoms += len(cluster.atoms)
                if return_clusters: 
                    fitting_list.append(cluster)
            else:
                if return_clusters: 
                    orphan_list.append(cluster)
        
        # fitness = (atoms placed / total atoms) / number of clusters
        score = (fitted_atoms / n_atoms) / n_clusters
        if return_clusters:
            return score, fitting_list, orphan_list
        return score

    # --- COARSE SCAN ---
    # use a large step to skip through empty space quickly

    # starting fitness score (how well the structure fits at a given z offset)
    best_coarse_score = -1.0
    best_coarse_z = 0
    # iterate over rough z offsets
    for z_offset in tqdm(range(0, int(z_box), coarse_step), desc="scan z offsets", leave=False):
        score = get_fitness_for_z(z_offset)
        if score > best_coarse_score:
            best_coarse_score = score
            best_coarse_z = z_offset

    # --- FINE SCAN ---
    # fine scan around the best z score from the previous scan
    
    # default return if nothing fits
    best_final_params = {
                "fitting_clusters": [],
                "orphan_clusters": [],
                "z_offset": 0,
                "fitness_score": -1.0
            }
    for z_offset in range(best_coarse_z - coarse_step, best_coarse_z + coarse_step, fine_step):
        score, fit_c, orph_c = get_fitness_for_z(z_offset, return_clusters=True)
        if score > best_final_params["fitness_score"]:
            best_final_params = {
                "fitting_clusters": fit_c,
                "orphan_clusters": orph_c,
                "z_offset": z_offset,
                "fitness_score": score
            }
    
    pbar.update(1)
    return best_final_params

def place_clusters(target_univ, clusters, z_offset):
    """
    Places clusters known to fit into the universe
    """
    if not clusters:
        return target_univ
    
    max_z = target_univ.dimensions[2]
    # Collect existing atoms and translated new atoms
    to_combine = [target_univ.atoms]
    for cluster in clusters:
        current_z = cluster.center_of_geometry()[2]
        # make sure the structure is set within box boundaries
        target_z = ((current_z + z_offset) % max_z)
        cluster.translate([0, 0, target_z - current_z])
        to_combine.append(cluster)
    
    # combine all clusters at once
    new_univ = mda.Merge(*to_combine)
    # restore original box
    new_univ.dimensions = target_univ.dimensions
    # translate overhanging atoms over periodic boundaries 
    new_univ.atoms.wrap(compound="atoms")
    return new_univ

def merge_structures(proteins_df):
    """
    Packs structures one-by-one, prioritizing high-mass, low-fragmentation placements.
    """
    # create a copy of the df to track progress
    placing_queue_df = proteins_df.copy()
    # setup merged universe
    # start with the one with the least clusters
    placing_queue_df["n_clusters"] = placing_queue_df["subaggregates"].apply(len)
    placing_queue_df = placing_queue_df.sort_values(by="n_clusters", ascending=True)
    first_struct = placing_queue_df.index[0]
    merged = placing_queue_df.loc[first_struct, "universe"].copy() 
    print(f"Initialized the structure with {first_struct} ({placing_queue_df.loc[first_struct, "n_clusters"]} aggregates).")

    # Remove the first structure from the queue
    placing_queue_df.drop(index=first_struct, inplace=True)

    # calculate box size of the new universe
    x_dim, y_dim, z_dim = calculate_box_from_condensate(proteins_df)
    # ***********************************************************************
    # Test
    x_dim, y_dim, z_dim = x_dim, y_dim, 2000
    # ***********************************************************************
    merged.dimensions = [x_dim, y_dim, z_dim, 90, 90, 90]
    print(f"New structure dimensions set at x: {x_dim}, y: {y_dim}, z: {z_dim}.")

    # center the first structure in the new big box
    center = merged.atoms.center_of_geometry()
    merged.atoms.translate([x_dim/2 - center[0], y_dim/2 - center[1], z_dim/2 - center[2]])
    merged.atoms.wrap(compound="atoms")

    # center all other structures the same way 
    for row in placing_queue_df.itertuples():
        u = row.universe
        u.dimensions = [x_dim, y_dim, z_dim, 90, 90, 90]
        center = u.atoms.center_of_geometry()
        u.atoms.translate([x_dim/2 - center[0], y_dim/2 - center[1], z_dim/2 - center[2]])
        u.atoms.wrap(compound="atoms")

    # place the the parts of the structures that fit in the new box one at a time
    # progress bar to pass the time
    pbar_outer = tqdm(total=placing_queue_df.shape[0], desc="placing structures", leave=True, unit="struct")
    # save aggregates that would overlap with the already placed structures
    all_orphan_clusters = []

    while not placing_queue_df.empty:
        pbar_inner = tqdm(total=placing_queue_df.shape[0], desc="explore structure fit", leave=False, unit="struct")
        # figure out which structure to add to the main universe next
        placing_queue_df["fitness_results"] = placing_queue_df.apply(
            lambda row: evaluate_structure_fitting(merged, row["subaggregates"], row["n_atoms"], pbar_inner), 
            axis=1
        )
        pbar_inner.close()
        # extract fitness score and sort for the best
        placing_queue_df["current_best_score"] = placing_queue_df["fitness_results"].apply(lambda x: x["fitness_score"])
        placing_queue_df = placing_queue_df.sort_values(by="current_best_score", ascending=False)
     
        # first row is to be placed in this iteration
        fittest_structure = placing_queue_df.index[0]
        fittest_params = placing_queue_df.loc[fittest_structure, "fitness_results"]
        merged = place_clusters(merged, fittest_params["fitting_clusters"], fittest_params["z_offset"])
        # save the clusters that couldn't be placed yet
        if fittest_params["orphan_clusters"]:
            all_orphan_clusters.extend(fittest_params["orphan_clusters"])
        
        # info
        n_fitting = len(fittest_params["fitting_clusters"])
        n_all = n_fitting + len(fittest_params["orphan_clusters"])
        abs_z = (fittest_params["z_offset"] + (z_dim/2)) % z_dim
        pbar_outer.write(f"Placed {fittest_structure}: {n_fitting}/{n_all} clusters fit "
              f"around Z={abs_z} (Score: {fittest_params["fitness_score"]:.3f})")

        # removing the placed universe from the queue
        placing_queue_df.drop(index=fittest_structure, inplace=True)

        pbar_outer.update(1)

    pbar_outer.close()

    merged.atoms.wrap(compound="segments")

    return merged
