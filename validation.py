from pathlib import Path
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
from openmm import app, unit, XmlSerializer, Platform
import openmm as mm

def check_cluster_overlaps(u, proteins_df, cutoff=10.0):
    """
    Groups segments by their 2-character prefix and checks for 
    collisions between these groups (ignoring internal chain-chain contacts).
    """
    agg_selections = []
    for structure in proteins_df.itertuples():
        for subaggregate in structure.subaggregates:
            seg_list = subaggregate["segIDs"]
            selection = f"segid {' '.join(seg_list)}"
            # selection = f"segid {seg_list[0]}"
            # if len(seg_list) > 1:
            #     for seg_id in seg_list[1:]:
            #         selection += f" or segid {seg_id}" 
            agg_selections.append(selection)                    
    
    subaggregate_ags = [u.select_atoms(selection) for selection in agg_selections]
    n_clusters = len(subaggregate_ags)
    total_collisions = 0
    checked_pairs = []

    # 2. Nested loop to check only DIFFERENT structures
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i >= j: continue # Avoid double-counting and self-interaction
            
            group1 = subaggregate_ags[i]
            group2 = subaggregate_ags[j]
            
            # Use capped_distance for the two distinct AtomGroups
            pairs = capped_distance(group1.positions, 
                                    group2.positions, 
                                    max_cutoff=cutoff, 
                                    box=u.dimensions)
            
            count = len(pairs[0])
            if count > 0:
                print(f"Collision: cluster {i} <-> cluster {j} | {count} atoms overlapping")
            
            total_collisions += count
            
    return total_collisions

def order_structure(universe, order):
    required_order = order
    sorted_segments = []
    for seg_prefix in required_order:
        for seg_idx in range(1, 41):
            seg_id = f"{seg_prefix}{seg_idx:03d}"
            # select protein by protein in order of the original structures
            selection = universe.select_atoms(f"segid {seg_id}")
            
            if len(selection) > 0:
                sorted_segments.append(selection)
            else:
                print(f"Warning: No atoms found for {seg_id}") 
    
    ordered_univ = mda.Merge(*sorted_segments)
    ordered_univ.dimensions = universe.dimensions
    print("--- Universe particles ordered ---")
    return ordered_univ



def make_energy_sim(system_xml: Path, top_pdb: Path, T, platform_name="CPU", gamma=0.01, dt_ps=0.01):
    system = XmlSerializer.deserialize(system_xml.read_text())
    pdb = app.PDBFile(str(top_pdb))
    integ = mm.LangevinMiddleIntegrator(T*unit.kelvin, gamma/unit.picosecond, dt_ps*unit.picoseconds)
    platform = Platform.getPlatformByName(platform_name)
    sim = app.Simulation(pdb.topology, system, integ, platform)
    box = pdb.topology.getPeriodicBoxVectors()
    if box is not None:
        a,b,c = box
        sim.context.setPeriodicBoxVectors(a,b,c)
    return sim

def calculate_universe_epot(system_xml: Path, universe, pdb_path: Path, T=260.15):

    print("Creating simulation...")
    sim = make_energy_sim(
        system_xml=Path("inputs/all_comps_condensed_260.15K.xml"), 
        top_pdb=pdb_path, 
        T=T
    )
    mda_positions = universe.atoms.positions / 10.0
    sim.context.setPositions(mda_positions)

    print("Calculating potential energy...")
    st = sim.context.getState(getEnergy=True, getPositions=True)
    pot_energy = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    print(f"--- Energy calculation finished ---")

    return pot_energy

    