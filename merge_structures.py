import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import math

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
    1. Avoiding self-interaction (XY > 2*Rg of the largest protein)
    2. Ability of condensates to span the entire box in XY direction (XY << V(all particles)/sigma)
    3. Rough box shape (8*XY <= Z <= 10*XY)
    4. Overall protein concentration (V(box) ~ concentration)

    :param universe_dict: dictionary with {prot_name: universe, subaggregates, chain_size}
    :param sigmas: average size of beads (Angstroem)
    """
    # Constraint 1: 
    min_xy = 2 * proteins_df["rg_max"].max()

    # Constraint 2: 
    # volume is approximated by just adding the volumes of boxes surrounding a bead and adding them together
    max_xy = math.sqrt(sigmas**2 * proteins_df["n_atoms"].min())
    if max_xy < min_xy:
        max_xy = min_xy


    # Constraint 3: 
    min_z = min_xy * 8
    max_z = max_xy * 10

    # Constraint 4:
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

def evaluate_structure_fitting(target_univ, clusters, n_atoms):
    """
    Tests how well a structure fits in the current universe at different z-offsets.
    Returns the metrics needed for the Fitness Score.
    """
    # Cache the positions of atoms already in the box to speed up checks
    target_pos = target_univ.atoms.positions
    box = target_univ.dimensions

    # dict to save how much of the structure can fit
    best_fitness = 0
    # iterate over possible z offsets
    for z_offset in range(0, int(box[2]), 20):
        fitting_clusters = []
        orphan_clusters = []
        # iterate over all subaggregates in the structure to see if they would fit
        for cluster in clusters:
            fits = check_placement_feasibility(target_pos, cluster, z_offset, box)
            if fits:
                fitting_clusters.append(cluster)
            else:
                orphan_clusters.append(cluster)
        
        if len(fitting_clusters) > best_fitness:
            best_fitness = len(fitting_clusters)
            best_z_offset = z_offset
            best_fitting_clusters = fitting_clusters
            best_orphan_clusters = orphan_clusters
            
    # calculate how much of the structure is placed
    fit_atoms_ratio = sum(len(c.atoms) for c in best_fitting_clusters)/n_atoms
    summary_dict = {
        "fitting_clusters": best_fitting_clusters,
        "orphan_clusters": best_orphan_clusters, 
        "z_offset": best_z_offset
        }
    return fit_atoms_ratio, summary_dict

def place_clusters(target_univ, clusters, z_offset):
    """
    Places clusters known to fit into the universe
    """
    for cluster in clusters:
        cluster.translate([0, 0, z_offset])
        new_univ = mda.Merge(target_univ.atoms, cluster)
    new_univ.dimenstions = target_univ.dimensions
    target_univ = new_univ

def merge_structures(proteins_df):
    # setup merged universe

    # Start with the one with the least clusters
    proteins_df["n_clusters"] = proteins_df["subaggregates"].apply(len)
    df_sorted = proteins_df.sort_values(by="n_clusters", ascending=True)

    merged = df_sorted.iloc[0]["universe"].copy()
    # calculate box size of the new universe
    x_dim, y_dim, z_dim = calculate_box_from_condensate(proteins_df)
    merged.atoms.dimensions = [x_dim, y_dim, z_dim, 90, 90, 90]

    # center the first structure in the new big box
    current_c = merged.atoms.center_of_geometry()
    merged.atoms.translate([x_dim/2 - current_c[0], y_dim/2 - current_c[1], z_dim/2 - current_c[2]])

    # center all other structures the same way 
    for structure in df_sorted.iloc[1:]["universe"]:
        structure.atoms.dimensions = [x_dim, y_dim, z_dim, 90, 90, 90]
        current_c = merged.atoms.center_of_geometry()
        structure.atoms.translate([x_dim/2 - current_c[0], y_dim/2 - current_c[1], z_dim/2 - current_c[2]])

    # add all the other structures
    placing_queue_df = df_sorted.iloc[1:, :].copy()
    remaining_clusters = []

    while placing_queue_df.shape[0] > 0:
        # figure out which structure to add to the main universe next
        placing_queue_df["fit_atoms_ratio"], placing_queue_df["fitting_parameters"] = placing_queue_df.apply(lambda row: evaluate_structure_fitting(merged, row["subaggregates"], row["n_atoms"]), axis=1)
        # "fitness" describes: the less clusters in the structure the better and the more of the structure's atoms can be placed the better
        placing_queue_df["fitness_score"] = placing_queue_df["fit_atoms_ratio"] / placing_queue_df["n_clusters"]
        placing_queue_df = placing_queue_df.sort_values(by="fitness_score", ascending=False)
        
        # first row is to be placed in this iteration
        fitting_parameters = placing_queue_df.iloc[0]["fitting_parameters"]
        place_clusters(merged, fitting_parameters["fitting_clusters"], fitting_parameters["z_offset"])
        remaining_clusters.append(fitting_parameters["orphan_clusters"])

        # removing the placed universe from the queue
        placing_queue_df.drop(index=0, inplace=True)

    return merged