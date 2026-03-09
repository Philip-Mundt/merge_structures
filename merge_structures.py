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

def evaluate_structure_fitting(target_univ, clusters, n_atoms, v=False):
    """
    Evaluates Z-offsets and picks the best one based on atom-ratio/cluster-count.
    Returns a single dictionary containing the winning parameters.
    """
    # Cache the positions of atoms already in the box to speed up checks
    target_pos = target_univ.atoms.positions
    box = target_univ.dimensions
    z_box = int(box[2])
    total_n_clusters = len(clusters)

    # starting fitness score (how well the structure fits at a given z offset)
    best_fitness_score = -1.0
    # fefault return if nothing fits
    best_parameters = {
        "fitting_clusters": [],
        "orphan_clusters": clusters,
        "z_offset": 0,
        "fitness_score": 0.0
    }
    # iterate over possible z offsets
    for z_offset in tqdm(range(0, z_box, 50), desc="Testing z_offsets...", leave=False, unit="step"):
        if v:
            print(f"Testing z-offset: {z_offset}")
            print(f"Clusters to test: {len(clusters)}")
        fitting_clusters = []
        orphan_clusters = []
        # iterate over all subaggregates in the structure to see if they would fit
        for cluster in clusters:
            # to tackle performance issues: only check for overlaps in same z-slice as cluster
            c_pos = cluster.positions
            z_min_query = c_pos[:, 2].min() + z_offset - 15.0
            z_max_query = c_pos[:, 2].max() + z_offset + 15.0
            mask = (target_pos[:, 2] >= z_min_query) & (target_pos[:, 2] <= z_max_query)
            local_target_pos = target_pos[mask]

            # shortcut if z-slice is empty
            if len(local_target_pos) == 0:
                fitting_clusters.append(cluster)
            else:
                fits = check_placement_feasibility(local_target_pos, cluster, z_offset, box)
                if fits:
                    fitting_clusters.append(cluster)
                else:
                    orphan_clusters.append(cluster)
        if v:
            print(f"Number of fitting clusters: {len(fitting_clusters)}")
        # calculate fitness for structure at specific z-offset
        fit_atoms_count = sum(len(c.atoms) for c in fitting_clusters)
        fit_atoms_ratio = fit_atoms_count / n_atoms
        # fitness = placed mass (%) / # of clusters in the structure 
        # ensures big clusters are placed first so they find place in the box
        fitness_score = fit_atoms_ratio / total_n_clusters 


        if fitness_score > best_fitness_score:
            best_fitness_score = fitness_score
            best_parameters = {
                "fitting_clusters": fitting_clusters,
                "orphan_clusters": orphan_clusters,
                "z_offset": z_offset,
                "fitness_score": fitness_score
            }
                
    return best_parameters

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
        c = cluster.center_of_geometry()
        if (c[2] + z_offset) > max_z:
            cluster_offset = z_offset - max_z
        else:
            cluster_offset = z_offset
        cluster.translate([0, 0, cluster_offset])
        to_combine.append(cluster)
    
    # combine all clusters at once
    new_univ = mda.Merge(*to_combine)
    # restore original box
    new_univ.dimensions = target_univ.dimensions
    return new_univ

def merge_structures(proteins_df, v=False):
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
    if v:
        print(f"Initialized the structure with the structure {first_idx} ({placing_queue_df.loc[first_idx, "n_clusters"]} clusters).")

    # calculate box size of the new universe
    x_dim, y_dim, z_dim = calculate_box_from_condensate(proteins_df)
    merged.dimensions = [x_dim, y_dim, z_dim, 90, 90, 90]
    if v:
        print(f"New universe dimensions set at [{x_dim}, {y_dim}, {z_dim}].")

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
    pbar_outer = tqdm(total=placing_queue_df.shape[0], desc="Placing Structures...", unit="struct")

    while not placing_queue_df.empty:
        if v:
            print(f"Structures yet to set: {placing_queue_df.index}")
        # figure out which structure to add to the main universe next
        placing_queue_df["fitness_results"] = placing_queue_df.apply(
            lambda row: evaluate_structure_fitting(merged, row["subaggregates"], row["n_atoms"], v), 
            axis=1
        )
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
        
        print(f"Placed {fittest_idx}: {len(fittest_structure['fitting_clusters'])} clusters fit "
              f"at Z={fittest_structure['z_offset']} (Score: {fittest_structure['fitness_score']:.3f})")

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