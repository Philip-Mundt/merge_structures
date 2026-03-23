import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def assign_segment_ids(universe, protein_name, existing_segids={}):
    """
    Assigns unique 4-character segment IDs to protein chains.
    
    Format: Single character [A-Z, a-z][Chain Number] (e.g., A001, F641)
    Handles cases where chain IDs repeat in input PDBs.
    
    Args:
        universe: MDAnalysis Universe object
        protein_name: Protein identifier (e.g., "hpl2", "lin13")
        existing_segids: External counter dict to maintain uniqueness across files -> {protein: (prefix, nmb)}
    """
    prefix_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    if protein_name in existing_segids.keys():
        prefix = existing_segids[protein_name][0]
        chain_count = existing_segids[protein_name][1] + 1
    else:
        prefix = prefix_chars[len(existing_segids)]
        chain_count = 1

    a = universe.atoms


    # Track the start position if a new chain
    chain_start = 0

    for i in range(1, len(a)):
        last = a[i-1]
        curr = a[i]
        
        if last.chainID != curr.chainID:
            # Slice based on positions in the current AtomGroup
            current_seg = a[chain_start:i]
            
            seg_id = f"{prefix}{chain_count:03d}"
            # create new segment
            new_seg = universe.add_Segment(segid=seg_id)

            # assign residues to this segment
            current_seg.residues.segments = new_seg

            # Update for next chain
            chain_start = i
            chain_count += 1

    # Get the very last protein chain that the loop finishes on
    last_seg = a[chain_start:]
    seg_id = f"{prefix}{chain_count:03d}"
    new_seg = universe.add_Segment(segid=seg_id)
    last_seg.residues.segments = new_seg

    existing_segids[protein_name] = (prefix, chain_count)
    
    # rebuild universe to clean empty segments
    universe = mda.Merge(a)

    return universe, prefix

def identify_subaggregates(universe, cutoff=15.0):
    """
    Clusters proteins if any bead of one is within 'cutoff' of any bead of another.
    """
    # 1. Collect all non-empty protein groups
    segments = [s for s in universe.segments if len(s.atoms) > 0]
    prots = [s.atoms for s in universe.segments if len(s.atoms) > 0]
    n_prots = len(prots)
    if n_prots == 0: return []

    # 2. Build Adjacency Matrix
    # Start with an identity matrix (every protein is connected to itself)
    adj = np.eye(n_prots, dtype=bool)

    # 3. Double-loop to find contacts between protein pairs
    box = universe.dimensions
    for i in range(n_prots):
        for j in range(i + 1, n_prots):
            # capped_distance returns pairs of indices (dist < cutoff)
            # if the returned array is not empty, the proteins are touching
            pairs = capped_distance(prots[i].positions, 
                                    prots[j].positions, 
                                    max_cutoff=cutoff, 
                                    box=box)
            
            if len(pairs[0]) > 0:
                adj[i, j] = adj[j, i] = True

    # 4. Find connected components
    graph = csr_matrix(adj)
    n_clusters, labels = connected_components(graph)

    # 5. Group into AtomGroups & calculate cylinders
    xdim, ydim, zdim = box[:3]
    subaggregates = []
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]

        seg_list = [segments[idx].segid for idx in cluster_indices]
        # Summing AtomGroups to create a single combined AtomGroup
        cluster_ag = mda.Merge(*[prots[idx] for idx in cluster_indices]).atoms
        # to calculate the center of the aggregate, it has to be unwrapped along periodic boundaries
        # to unwrap the aggregate, the shortest distance of all points to a random reference point
        # in the aggregate is calculated and the atoms arranged around this point
        # disclaimer: works only if the aggregate spans less then half of the respective box dimension
        # (otherwise the zylinder might span a lot of empty space)        
        pos = cluster_ag.positions.copy()
        ref_pos = pos[0]
        dp = pos - ref_pos
        dp[:, 0] -= xdim * np.round(dp[:, 0] / xdim)
        dp[:, 1] -= ydim * np.round(dp[:, 1] / ydim)
        dp[:, 2] -= zdim * np.round(dp[:, 2] / zdim)
        unwrapped_pos = ref_pos + dp

        # Cylinder dimensions
        z_min = unwrapped_pos[:, 2].min()
        z_max = unwrapped_pos[:, 2].max()
        height = z_max - z_min
        center = np.array([
            unwrapped_pos[:, 0].mean(axis=0),
            unwrapped_pos[:, 1].mean(axis=0),
            (z_min + z_max)/2
        ])
        radius = np.sqrt(np.sum((unwrapped_pos[:, :2] - center[:2])**2, axis=1)).max()

        subaggregates.append({
            "cluster": cluster_ag,
            "segIDs": seg_list,
            "center": center,
            "radius": radius,
            "height": height
        })

    return subaggregates