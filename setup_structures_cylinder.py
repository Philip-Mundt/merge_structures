import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

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
        # Summing AtomGroups to create a single combined AtomGroup
        cluster_ag = sum(prots[idx] for idx in cluster_indices)

        # find true center respecting periodic boundaries
        pos = cluster_ag.positions.copy()
        ref_pos = pos[0]
        dp = pos - ref_pos
        # Minimum Image Convention for unwrapping
        dp[:, 0] -= xdim * np.round(dp[:, 0] / xdim)
        dp[:, 1] -= ydim * np.round(dp[:, 1] / ydim)
        dp[:, 2] -= zdim * np.round(dp[:, 2] / zdim)
        
        unwrapped_pos = ref_pos + dp
        # Cylinder dimensions
        center = np.array([
            unwrapped_pos[0, :].mean(axis=0),
            unwrapped_pos[1, :].mean(axis=0),
            (unwrapped_pos[2, :].min() + unwrapped_pos[2, :].max())/2
        ])
        height = unwrapped_pos[:, 2].max() - unwrapped_pos[:, 2].min()
        radius = np.sqrt(np.sum((unwrapped_pos[:, :2] - center[:2])**2, axis=1)).max()

        subaggregates.append({
            "cluster": cluster_ag,
            "center": [center[0] % xdim, center[1] % ydim, center[2] % zdim],
            "radius": radius,
            "height": height
        })

    return subaggregates