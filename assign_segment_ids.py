def assign_segment_ids(universe, protein_name):
    """
    """
    # create a 2 digit short (first character + first number)
    first_letter = protein_name[0]
    first_digit = next(c for c in protein_name if c.isdigit())
    prefix = first_letter + first_digit

    i = 0
    for segment in universe.segments:
        i += 1 
        for atom in segment.atoms:
            atom.segid = prefix + str(i).rjust(2, "0")