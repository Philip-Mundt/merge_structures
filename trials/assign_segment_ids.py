def assign_segment_ids(universe, protein_name):
    """
    Assigns unique 4-character segment IDs to protein chains.
    
    Format: [Protein Type][Chain Number] (e.g., H201, L101)
    Handles cases where chain IDs repeat in input PDBs.
    
    Args:
        universe: MDAnalysis Universe object
        protein_name: Protein identifier (e.g., "hpl2", "lin13")
        segid_counter: External counter dict to maintain uniqueness across files
    """
    # create a 2 digit short (first character + first number)
    first_letter = protein_name[0]
    if any(protein_name.isdigit()):
        first_digit = next(c for c in protein_name if c.isdigit())
        prefix = first_letter + first_digit
    else: 
        prefix = protein_name[0:2]
    

    a = universe.atoms

    chain_count = 1
    # Track the start position if a new chain
    chain_start = 0


    for i in range(1, len(a)):
        last = a[i-1]
        curr = a[i]
        
        if last.chainID != curr.chainID:
            # Slice based on positions in the current AtomGroup
            current_seg = a[chain_start:i]
            
            # Create unique segment name
            seg_id = prefix + str(chain_count).rjust(2, "0")

            # update segment ID in the metadata
            current_seg.atoms.segids = seg_id

            # Update for next chain
            chain_start = i
            chain_count += 1

    # Get the very last protein chain that the loop finishes on
    last_seg = a[chain_start:]
    seg_id = prefix + str(chain_count).rjust(2, "0")
    last_seg.segids = seg_id