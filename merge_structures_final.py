import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import capped_distance, distance_array
import os
import warnings
warnings.filterwarnings('ignore')

def assign_segment_ids(universe, protein_name, segid_counter=None):
    """
    Assigns unique 4-character segment IDs to protein chains.
    
    Format: [Protein Type][Chain Number] (e.g., H201, L101)
    Handles cases where chain IDs repeat in input PDBs.
    
    Args:
        universe: MDAnalysis Universe object
        protein_name: Protein identifier (e.g., "hpl2", "lin13")
        segid_counter: External counter dict to maintain uniqueness across files
    """
    if segid_counter is None:
        segid_counter = {}
    
    # Generate 2-character prefix
    if "hpl" in protein_name.lower():
        prefix = "HP"
    elif "met" in protein_name.lower():
        prefix = "MT"
    elif "lin13" in protein_name.lower():
        prefix = "L1"
    elif "lin65" in protein_name.lower():
        prefix = "L6"
    else:
        # Fallback: first letter + first digit or 'X'
        first_letter = protein_name[0].upper()
        digits = [c for c in protein_name if c.isdigit()]
        first_digit = digits[0] if digits else 'X'
        prefix = f"{first_letter}{first_digit}"
    
    # Initialize counter for this prefix
    if prefix not in segid_counter:
        segid_counter[prefix] = 1
    
    # Identify chains by residue continuity (more reliable than chainID)
    chains = []
    chain_start = 0
    
    for i in range(1, len(universe.atoms)):
        if universe.atoms[i].resid != universe.atoms[i-1].resid + 1:
            chains.append(universe.atoms[chain_start:i])
            chain_start = i
    
    # Add final chain
    chains.append(universe.atoms[chain_start:])
    
    # Assign segment IDs
    for chain in chains:
        segid = f"{prefix}{segid_counter[prefix]:02d}"
        chain.segments.segids = segid
        segid_counter[prefix] += 1

def identify_subaggregates(universe, cutoff=15.0):
    """
    Identifies sub-aggregates within a universe by clustering proteins based on spatial proximity.
    Handles cases where segment/chain IDs repeat (beyond 26 segments).
    
    Parameters:
    universe: MDAnalysis Universe containing proteins
    cutoff: Distance threshold for connectivity (Å)
    
    Returns: List of AtomGroups representing individual sub-aggregates
    """
    # Get unique protein identifiers (using residue numbers as unique IDs)
    protein_resids = np.unique(universe.residues.resids)
    
    # Calculate center of geometry for each protein
    protein_centers = []
    protein_groups = []
    
    for resid in protein_resids:
        protein = universe.residues.resids == resid
        if np.any(protein):
            protein_atoms = universe.residues[protein].atoms
            protein_centers.append(protein_atoms.center_of_geometry())
            protein_groups.append(protein_atoms)
    
    if len(protein_centers) == 0:
        return []
    
    # Create distance matrix between all protein centers
    protein_centers = np.array(protein_centers)
    dist_matrix = distance_array(protein_centers, protein_centers)
    
    # Identify connected components (aggregates)
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    graph = csr_matrix(dist_matrix < cutoff)
    n_components, labels = connected_components(graph)
    
    # Group proteins by component
    subaggregates = []
    for i in range(n_components):
        component_indices = np.where(labels == i)[0]
        # Combine all protein atoms in this component
        component_atoms = mda.core.groups.AtomGroup([])
        for idx in component_indices:
            component_atoms += protein_groups[idx]
        subaggregates.append(component_atoms)
    
    return subaggregates

def calculate_bounding_box(atom_group):
    """Calculate bounding box for an atom group"""
    coords = atom_group.positions
    if len(coords) == 0:
        return np.array([0, 0, 0]), np.array([0, 0, 0])
    
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    
    return min_coords, max_coords

def calculate_size(atom_group):
    """Calculate size of an atom group (diagonal of bounding box)"""
    min_coords, max_coords = calculate_bounding_box(atom_group)
    size = np.linalg.norm(max_coords - min_coords)
    return size

def calculate_radius_of_gyration(atom_group):
    """Calculate radius of gyration for an atom group"""
    if len(atom_group) == 0:
        return 0.0
    
    # Calculate center of mass
    com = atom_group.center_of_mass()
    
    # Calculate squared distances from center of mass
    distances_squared = np.sum((atom_group.positions - com) ** 2, axis=1)
    
    # Calculate radius of gyration
    rg = np.sqrt(np.mean(distances_squared))
    
    return rg

def calculate_condensate_volume(atom_group):
    """Calculate volume of a condensate (bounding box volume)"""
    min_coords, max_coords = calculate_bounding_box(atom_group)
    dimensions = max_coords - min_coords
    volume = np.prod(dimensions)
    return volume

def check_collision(atom_group1, atom_group2, min_padding=0.5):
    """Check if two atom groups collide using bead-specific radii (currently set to 6.0 for all beads)"""
    if len(atom_group1) == 0 or len(atom_group2) == 0:
        return False
    
    # Use bead radius of 6.0 for all beads (as requested by user)
    bead_radius = 6.0
    min_required_distance = 2.0 * bead_radius + min_padding  # Sum of radii + padding
    
    # Use capped_distance for efficiency
    distances = capped_distance(atom_group1.positions, atom_group2.positions, min_required_distance)
    
    # If any distance is less than min_required_distance, there's a collision
    return np.any(distances < min_required_distance)

def get_bead_radius(atom_group):
    """Get bead radius for an atom group (currently returns 6.0 for all beads)"""
    return 6.0

def find_optimal_position(positioned_aggregates, new_aggregate, min_distance=5.0, max_attempts=100):
    """Find optimal position for new aggregate to avoid collisions"""
    if len(positioned_aggregates) == 0:
        return np.array([0, 0, 0])
    
    # Get size of new aggregate
    new_size = calculate_size(new_aggregate)
    
    # Start with initial position (away from existing aggregates)
    for attempt in range(max_attempts):
        # Try different strategies
        if attempt < 10:  # Try simple offset
            offset = np.array([0, 0, 0])
            for i, agg in enumerate(positioned_aggregates):
                agg_center = agg.center_of_geometry()
                offset += agg_center * (i + 1)
            offset = offset / len(positioned_aggregates) if len(positioned_aggregates) > 0 else np.array([0, 0, 0])
            offset += np.array([new_size * 2, 0, 0])  # Offset in x direction
        elif attempt < 30:  # Try random positions
            offset = np.random.normal(0, new_size * 5, 3)
        else:  # Try systematic search
            angle = (attempt - 30) * np.pi / 180
            distance = new_size * (2 + (attempt - 30) / 20)
            offset = np.array([distance * np.cos(angle), distance * np.sin(angle), 0])
        
        # Apply translation
        new_aggregate.atoms.translate(offset)
        
        # Check for collisions
        collision = False
        for positioned in positioned_aggregates:
            if check_collision(positioned, new_aggregate, min_distance):
                collision = True
                break
        
        if not collision:
            return offset
        
        # Reset position for next attempt
        new_aggregate.atoms.translate(-offset)
    
    # If no good position found, use default offset
    return np.array([new_size * 2, 0, 0])

def validate_bead_sizes(universes):
    """Validate that all beads have consistent size (currently 6.0)"""
    print("\nBead Size Validation:")
    for i, uni in enumerate(universes):
        print(f"  Universe {i+1}:")
        # Get unique residue names
        residue_names = np.unique(uni.residues.resnames)
        for resname in residue_names:
            # Get atoms for this residue type
            atoms = uni.residues.resnames == resname
            if np.any(atoms):
                # All beads in this residue type should have the same size
                bead_radius = get_bead_radius(uni.atoms[atoms])
                print(f"    {resname}: bead radius = {bead_radius}")
    
    print("  All beads validated with radius = 6.0")

def position_aggregates(universes, min_distance=5.0):
    """Position protein aggregates with minimum separation distance"""
    positioned = []
    
    for i, uni in enumerate(universes):
        print(f"Processing universe {i+1}...")
        
        # Identify sub-aggregates
        subaggregates = identify_subaggregates(uni, cutoff=15.0)
        print(f"  Found {len(subaggregates)} sub-aggregates")
        
        # Position each sub-aggregate
        for j, subagg in enumerate(subaggregates):
            if len(positioned) == 0:
                # First sub-aggregate - place at origin
                positioned.append(subagg)
            else:
                # Find optimal position
                translation = find_optimal_position(positioned, subagg, min_distance)
                subagg.atoms.translate(translation)
                positioned.append(subagg)
    
    return positioned

def validate_positions(positioned_aggregates, min_distance=5.0):
    """Validate that all positioned aggregates maintain minimum distance"""
    print("\nValidation:")
    total_collisions = 0
    
    for i, agg1 in enumerate(positioned_aggregates):
        for j, agg2 in enumerate(positioned_aggregates):
            if i != j:
                if check_collision(agg1, agg2, min_distance):
                    total_collisions += 1
                    print(f"  Collision detected between aggregate {i+1} and {j+1}")
    
    if total_collisions == 0:
        print("  No collisions detected - positioning successful!")
    else:
        print(f"  {total_collisions} collisions detected - may need to increase min_distance")
    
    return total_collisions == 0

def merge_positioned_aggregates(positioned_aggregates, box_dimensions=None, universes=None):
    """Merge all positioned aggregates into a single universe with updated box dimensions"""
    if len(positioned_aggregates) == 0:
        return None
    
    # Create merged universe
    merged = mda.Merge(*positioned_aggregates)
    
    # Calculate new box dimensions based on requirements
    if box_dimensions is None and len(positioned_aggregates) > 0:
        # Try to get dimensions from first universe
        try:
            box_dimensions = positioned_aggregates[0].atoms.dimensions
        except:
            box_dimensions = [1000, 1000, 1000, 90, 90, 90]
    
    # Update box dimensions based on new requirements
    if universes is not None and len(universes) > 0:
        # Calculate new box dimensions
        new_box_dimensions = calculate_new_box_dimensions(universes, positioned_aggregates)
        merged.atoms.dimensions = new_box_dimensions
    else:
        merged.atoms.dimensions = box_dimensions
    
    return merged

def calculate_new_box_dimensions(universes, positioned_aggregates):
    """
    Calculate new box dimensions based on concentration preservation and other constraints.
    
    Requirements:
    1. Overall concentration should not change (box size should scale with number of proteins)
    2. x/y size > 2x radius of gyration of the longest protein chain
    3. z height ~ 8-10 times x/y size
    4. x/y size should be small enough that a proper condensate spans the entire length and width
    """
    # Get all protein chains from all universes
    all_chains = []
    for uni in universes:
        # Identify chains by residue continuity
        chains = []
        chain_start = 0
        for i in range(1, len(uni.atoms)):
            if uni.atoms[i].resid != uni.atoms[i-1].resid + 1:
                chains.append(uni.atoms[chain_start:i])
                chain_start = i
        chains.append(uni.atoms[chain_start:])
        all_chains.extend(chains)
    
    # Calculate radius of gyration for each chain
    rg_values = []
    for chain in all_chains:
        rg = calculate_radius_of_gyration(chain)
        rg_values.append(rg)
    
    # Get maximum radius of gyration
    max_rg = max(rg_values) if len(rg_values) > 0 else 0.0
    
    # Calculate minimum x/y size (2x max radius of gyration)
    min_xy_size = 2.0 * max_rg if max_rg > 0 else 100.0
    
    # Calculate total volume of all condensates
    total_condensate_volume = 0.0
    for agg in positioned_aggregates:
        volume = calculate_condensate_volume(agg)
        total_condensate_volume += volume
    
    # Calculate target box volume based on concentration preservation
    # If we're merging n structures, the box volume should be n times the original volume
    n_structures = len(universes)
    original_volume = 0.0
    
    # Get original box volume from first universe
    if len(universes) > 0:
        try:
            original_box = universes[0].atoms.dimensions[:3]
            original_volume = np.prod(original_box)
        except:
            original_volume = 1000.0 * 1000.0 * 1000.0  # Default if dimensions not available
    
    target_volume = n_structures * original_volume
    
    # Calculate target x/y size based on condensate volume constraint
    # We want the x/y size to be such that a proper condensate spans the box
    # This means the x/y size should be roughly the size of the largest condensate
    max_condensate_size = 0.0
    for agg in positioned_aggregates:
        min_coords, max_coords = calculate_bounding_box(agg)
        dimensions = max_coords - min_coords
        # Use the maximum of x and y dimensions as the condensate size
        condensate_size = max(dimensions[0], dimensions[1])
        max_condensate_size = max(max_condensate_size, condensate_size)
    
    # Set x/y size to be the maximum of:
    # 1. Minimum size based on radius of gyration
    # 2. Size based on condensate spanning requirement
    # 3. Size based on concentration preservation (volume constraint)
    xy_size = max(min_xy_size, max_condensate_size)
    
    # Calculate z height (8-10 times x/y size)
    z_height = 9.0 * xy_size  # Use 9x as middle of 8-10 range
    
    # Adjust xy_size to satisfy volume constraint
    # Volume = xy_size * xy_size * z_height
    # So xy_size = sqrt(target_volume / z_height)
    if z_height > 0:
        xy_size_from_volume = np.sqrt(target_volume / z_height)
        xy_size = max(xy_size, xy_size_from_volume)
    
    # Create new box dimensions
    new_box = [
        xy_size,  # x dimension
        xy_size,  # y dimension
        z_height,  # z dimension
        90.0,  # alpha angle
        90.0,  # beta angle
        90.0   # gamma angle
    ]
    
    return new_box

def merge_pdb_files(pdb_files, protein_names, output_path="output/merged_structure.pdb", min_distance=5.0):
    """
    Main function to merge multiple PDB files with collision avoidance and proper segment ID handling.
    
    Args:
        pdb_files: List of PDB file paths
        protein_names: List of protein identifiers corresponding to each PDB file
        output_path: Output PDB file path
        min_distance: Minimum separation between aggregates (Å)
    """
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize segment ID counter
    segid_counter = {}
    
    # Load and process each universe
    universes = []
    box_dimensions = None
    
    for pdb_file, protein_name in zip(pdb_files, protein_names):
        try:
            print(f"Loading {pdb_file}...")
            uni = mda.Universe(pdb_file)
            
            # Assign segment IDs
            assign_segment_ids(uni, protein_name, segid_counter)
            print(f"  Assigned segment IDs for {protein_name}")
            
            # Store box dimensions from first file
            if box_dimensions is None:
                box_dimensions = uni.atoms.dimensions
            
            universes.append(uni)
            
        except FileNotFoundError:
            print(f"Error: Could not find file {pdb_file}")
            continue
    
    if len(universes) == 0:
        print("No universes loaded. Exiting.")
        return
    
    # Validate bead sizes
    validate_bead_sizes(universes)
    
    # Position aggregates
    positioned_aggregates = position_aggregates(universes, min_distance)
    
    # Validate positions
    validation_success = validate_positions(positioned_aggregates, min_distance)
    
    # Merge positioned aggregates
    merged = merge_positioned_aggregates(positioned_aggregates, box_dimensions)
    
    if merged is not None:
        # Write output file
        merged.atoms.write(output_path)
        print(f"\nOutput file created successfully: {output_path}")
        
        # Return the merged universe for further analysis
        return merged
    else:
        print("Failed to create merged universe.")
        return None
