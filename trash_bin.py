def create_occupancy_grid(universe, box, rg_min):
    step_size = 2 * rg_min
    print(f"Step size rule {step_size}")
    grid_steps = [int(dim / step_size) for dim in box[:3]]
    step_sizes = [box[i] / grid_steps[i] for i in range(3)]

    grid = np.zeros(shape=tuple(grid_steps))
    print(f"Grid initialized with dimensions {grid_steps[0]}, {grid_steps[1]}, {grid_steps[2]}. Step sizes: {step_sizes[0]}, {step_sizes[1]}, {step_sizes[2]}")
    
    pos = universe.atoms.positions.copy()
    # wrap atom positions around periodic boundaries for the grid creation
    pos[:, 0] %= box[0]
    pos[:, 1] %= box[1]
    pos[:, 2] %= box[2]

    for z_step in range(grid_steps[2]):
        z_min = z_step * step_sizes[2]
        z_max = (z_step + 1) * step_sizes[2]
        if np.any((z_min <= pos[:, 2]) & (pos[:, 2] <= z_max)):
            for x_step in range(grid_steps[0]):
                x_min = x_step * step_sizes[0]
                x_max = (x_step + 1) * step_sizes[0] 
                if np.any((x_min <= pos[:, 0]) & (pos[:, 0] <= x_max)):
                    for y_step in range(grid_steps[1]):
                        y_min = y_step * step_sizes[1]
                        y_max = (y_step + 1) * step_sizes[1] 
                        if np.any((y_min <= pos[:, 1]) & (pos[:, 1] <= y_max)):
                            grid[x_step, y_step, z_step] = 1
    
    return grid
