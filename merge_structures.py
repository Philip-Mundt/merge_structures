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

def check_placement_feasibility(target_positions, cluster, test_z, box_dims):
    """
    Calculates if a cluster fits at a specific Z-offset WITHOUT moving it.
    
    :param target_positions: The (N, 3) array of atoms already in the master box.
    :param cluster: The AtomGroup we are testing.
    :param test_z: The Z-translation we are 'trying out'.
    :return: Boolean (True if it fits)
    """
    # 1. Create a temporary copy of the positions for math only
    # .positions returns a copy of the coordinates as a NumPy array
    trial_positions = cluster.positions.copy()
    
    # 2. Apply the Z-offset to the temporary array
    # [:, 2] accesses the Z-column of all atoms in this cluster
    trial_positions[:, 2] += test_z
    
    # 3. Check for overlaps against the target_positions
    # We pass our trial_positions array instead of the cluster itself
    pairs = capped_distance(target_positions, trial_positions, 
                            max_cutoff=10.0, box=box_dims)
    
    # If len(pairs[0]) is 0, no atoms are within 10A of each other
    return len(pairs[0]) == 0

def evaluate_structure_fitting(target_univ, clusters, n_atoms, pbar, coarse_step=150, fine_step=20):
    """
    Finds the best Z-offset for a structure using a two-pass search:
    1. Coarse scan: Quickly find the most promising region of the box.
    2. Fine tuning: Zoom in on that region for precision placement.
    """
    # Cache the positions of atoms already in the box to speed up checks
    target_pos = target_univ.atoms.positions
    box = target_univ.dimensions
    z_box = int(box[2])
    n_clusters = len(clusters)

    # --- COARSE SCAN ---
    # use a large step to skip through empty space quickly

    # starting fitness score (how well the structure fits at a given z offset)
    best_coarse_fitness = -1.0
    best_coarse_z = 0

    # iterate over rough z offsets
    for z_offset in tqdm(range(0, z_box, coarse_step), desc="scan z offsets", leave=False):
        fitted_atoms = 0
        # iterate over all subaggregates in the structure to see if they would fit
        for cluster in clusters:
            c_pos = cluster.positions
            # only check for overlaps in the Z-slice where the cluster will actually sit
            z_min, z_max = c_pos[:, 2].min() + z_offset - 15.0, c_pos[:, 2].max() + z_offset + 15.0
            mask = (target_pos[:, 2] >= z_min) & (target_pos[:, 2] <= z_max)
            
            # if slice is empty or no collision found
            if len(target_pos[mask]) == 0 or check_placement_feasibility(target_pos[mask], cluster, z_offset, box):
                fitted_atoms += len(cluster.atoms)
        
        # fitness = (mass placed / total mass) / number of clusters
        current_fitness = (fitted_atoms / n_atoms) / n_clusters
        
        # ensures big clusters that fit are placed first
        if current_fitness > best_coarse_fitness:
            best_coarse_fitness = current_fitness
            best_coarse_z = z_offset

    # --- FINE SCAN ---
    # fine scan around the best z score from the previous scan
    fine_start = best_coarse_z - coarse_step
    fine_end = best_coarse_z + coarse_step
    
    # default return if nothing fits
    best_final_params = {
        "fitting_clusters": [],
        "orphan_clusters": clusters,
        "z_offset": 0,
        "fitness_score": 0.0
    }

    for z_offset in range(fine_start, fine_end, fine_step):
        # this time: save the actual clusters that fit
        current_fitting = []
        current_orphans = []
        
        for cluster in clusters:
            c_pos = cluster.positions
            z_min, z_max = c_pos[:, 2].min() + z_offset - 15.0, c_pos[:, 2].max() + z_offset + 15.0
            mask = (target_pos[:, 2] >= z_min) & (target_pos[:, 2] <= z_max)
            
            if len(target_pos[mask]) == 0 or check_placement_feasibility(target_pos[mask], cluster, z_offset, box):
                current_fitting.append(cluster)
            else:
                current_orphans.append(cluster)
        
        fit_ratio = sum(len(c.atoms) for c in current_fitting) / n_atoms
        final_score = fit_ratio / n_clusters
        
        # create dictionary for outputs
        if final_score > best_final_params["fitness_score"]:
            best_final_params = {
                "fitting_clusters": current_fitting,
                "orphan_clusters": current_orphans,
                "z_offset": z_offset,
                "fitness_score": final_score
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
    return new_univ

def merge_structures(proteins_df):
    """
    Packs structures one-by-one, prioritizing high-mass, low-fragmentation placements.
    """
    # Create a copy to track progress
    placing_queue_df = proteins_df.copy()

    # setup merged universe
    # start with the one with the least clusters
    placing_queue_df["n_clusters"] = placing_queue_df["subaggregates"].apply(len)
    placing_queue_df = placing_queue_df.sort_values(by="n_clusters", ascending=True)
    first_idx = placing_queue_df.index[0]
    merged = placing_queue_df.loc[first_idx, "universe"].copy()    
    print(f"Initialized the structure with {first_idx} ({placing_queue_df.loc[first_idx, "n_clusters"]} clusters).")

    # calculate box size of the new universe
    x_dim, y_dim, z_dim = calculate_box_from_condensate(proteins_df)
    merged.dimensions = [x_dim, y_dim, z_dim, 90, 90, 90]
    print(f"New structure dimensions set at x: {x_dim}, y: {y_dim}, z: {z_dim}.")

    # center the first structure in the new big box
    current_c = merged.atoms.center_of_geometry()
    merged.atoms.translate([x_dim/2 - current_c[0], y_dim/2 - current_c[1], z_dim/2 - current_c[2]])

    # center all other structures the same way 
    for row in placing_queue_df.itertuples():
        u = row.universe
        u.dimensions = [x_dim, y_dim, z_dim, 90, 90, 90]
        current_c = u.atoms.center_of_geometry()
        shift = [x_dim/2 - current_c[0], y_dim/2 - current_c[1], z_dim/2 - current_c[2]]
        for cluster in row.subaggregates:
            cluster.translate(shift)

    # Remove anchor from queue
    placing_queue_df.drop(index=first_idx, inplace=True)
    all_orphan_clusters = []

    # progress bar to pass the time
    pbar_outer = tqdm(total=placing_queue_df.shape[0], desc="placing structures", leave=True, unit="struct")

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
        fittest_idx = placing_queue_df.index[0]
        fittest_structure = placing_queue_df.loc[fittest_idx, "fitness_results"]
        merged = place_clusters(merged, fittest_structure["fitting_clusters"], fittest_structure["z_offset"])
        # save the clusters that couldn't be placed yet
        if fittest_structure["orphan_clusters"]:
            all_orphan_clusters.extend(fittest_structure["orphan_clusters"])
        
        # info
        n_fitting = len(fittest_structure["fitting_clusters"])
        n_all = n_fitting + len(fittest_structure["orphan_clusters"])
        pbar_outer.write(f"Placed {fittest_idx}: {n_fitting}/{n_all} clusters fit "
              f"at Z={fittest_structure["z_offset"]} (Score: {fittest_structure["fitness_score"]:.3f})")

        # removing the placed universe from the queue
        placing_queue_df.drop(index=fittest_idx, inplace=True)

        pbar_outer.update(1)

    pbar_outer.close()

    return merged

def scavenge_orphans(target_univ, orphans, v=False):
    """
    Attempts to place orphaned clusters into the 'vacuum' regions of the box.
    """
    if not orphans:
        return target_univ

    # 1. Sort orphans by size (descending) - big orphans are harder to place
    orphans = sorted(orphans, key=len, reverse=True)
    
    # 2. Identify the 'Slab Zone' to avoid it
    # We find the Z-envelope of the current atoms
    z_coords = target_univ.atoms.positions[:, 2]
    slab_z_min, slab_z_max = z_coords.min(), z_coords.max()
    
    box = target_univ.dimensions
    z_box_max = box[2]

    # 3. Create a prioritized list of Z-offsets
    # We want to check the "vacuum" areas (near 0 and near z_box_max) first
    vacuum_range = []
    slab_range = []
    
    for z in range(0, int(z_box_max), 20): # Finer 20A steps for orphans
        if z < (slab_z_min - 20) or z > (slab_z_max + 20):
            vacuum_range.append(z)
        else:
            slab_range.append(z)
    
    # Prioritized search list: Vacuums first, then Slab gaps as a fallback
    search_offsets = vacuum_range + slab_range

    final_merged = target_univ
    placed_count = 0

    for i, cluster in enumerate(orphans):
        found_home = False
        target_pos = final_merged.atoms.positions # Refresh positions each time
        
        for z_off in search_offsets:
            # Re-use your optimized slice-checking logic
            c_pos = cluster.positions
            z_min_q = c_pos[:, 2].min() + z_off - 15.0
            z_max_q = c_pos[:, 2].max() + z_off + 15.0
            
            mask = (target_pos[:, 2] >= z_min_q) & (target_pos[:, 2] <= z_max_q)
            local_target_pos = target_pos[mask]
            
            if len(local_target_pos) == 0 or check_placement_feasibility(local_target_pos, cluster, z_off, box):
                # Place it!
                final_merged = place_clusters(final_merged, [cluster], z_off)
                found_home = True
                placed_count += 1
                if v: print(f"Orphan {i} placed at Z={z_off}")
                break
        
        if not found_home and v:
            print(f"Orphan {i} ({len(cluster)} atoms) could not find a home.")

    print(f"Scavenger complete: Placed {placed_count}/{len(orphans)} orphans.")
    return final_merged