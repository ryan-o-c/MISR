#!/usr/bin/env python3
# z3_rectangle_solver.py
from z3 import Solver, Int, And, Or, Not, sat
import os
import re

def solve_rectangle_problem(n, adjacency_list):
    """Solve and display rectangle intersection constraints using Z3."""
    s = Solver()
    s.set("timeout", 30000)  # 30 seconds

    grid_size = 2 * n
    # declare variables with same names as in the earlier code
    x1 = [Int(f"x1_{i}") for i in range(n)]
    x2 = [Int(f"x2_{i}") for i in range(n)]
    y1 = [Int(f"y1_{i}") for i in range(n)]
    y2 = [Int(f"y2_{i}") for i in range(n)]

    # bounds and positive-area constraints
    for i in range(n):
        s.add(x1[i] >= 0,        x1[i] <= grid_size - 2)
        s.add(x2[i] >= 1,        x2[i] <= grid_size - 1)
        s.add(y1[i] >= 0,        y1[i] <= grid_size - 2)
        s.add(y2[i] >= 1,        y2[i] <= grid_size - 1)
        s.add(x1[i] <= x2[i] - 1)
        s.add(y1[i] <= y2[i] - 1)

    # pairwise constraints (preserve same logical formulation)
    for i in range(n):
        for j in range(i + 1, n):
            x_overlap = Or(
                And(x1[i] <= x1[j], x1[j] <= x2[i]),
                And(x1[i] <= x2[j], x2[j] <= x2[i]),
                And(x1[j] <= x1[i], x1[i] <= x2[j]),
                And(x1[j] <= x2[i], x2[i] <= x2[j])
            )
            y_overlap = Or(
                And(y1[i] <= y1[j], y1[j] <= y2[i]),
                And(y1[i] <= y2[j], y2[j] <= y2[i]),
                And(y1[j] <= y1[i], y1[i] <= y2[j]),
                And(y1[j] <= y2[i], y2[i] <= y2[j])
            )

            if j in adjacency_list.get(i, []):
                # must overlap both dims
                s.add(x_overlap)
                s.add(y_overlap)
            else:
                # cannot overlap in both dims simultaneously
                s.add(Or(Not(x_overlap), Not(y_overlap)))

    print(f"Solving for {n} rectangles with adjacency list: {adjacency_list}")
    if s.check() == sat:
        m = s.model()
        print("\nSolution found! Rectangle coordinates:")
        for i in range(n):
            print(f"  Rect {i}: x=[{m[x1[i]]}, {m[x2[i]]}], y=[{m[y1[i]]}, {m[y2[i]]}]")

        # verification
        print("\nOverlap verification:")
        for i in range(n):
            for j in range(i + 1, n):
                x1_i, x2_i = m[x1[i]].as_long(), m[x2[i]].as_long()
                y1_i, y2_i = m[y1[i]].as_long(), m[y2[i]].as_long()
                x1_j, x2_j = m[x1[j]].as_long(), m[x2[j]].as_long()
                y1_j, y2_j = m[y1[j]].as_long(), m[y2[j]].as_long()

                x_overlap = (x1_i <= x1_j <= x2_i) or (x1_i <= x2_j <= x2_i) or \
                            (x1_j <= x1_i <= x2_j) or (x1_j <= x2_i <= x2_j)
                y_overlap = (y1_i <= y1_j <= y2_i) or (y1_i <= y2_j <= y2_i) or \
                            (y1_j <= y1_i <= y2_j) or (y1_j <= y2_i <= y2_j)

                has_edge = j in adjacency_list.get(i, [])
                print(f"  R{i}-R{j}: x_overlap={x_overlap}, y_overlap={y_overlap}, "
                      f"edge_expected={has_edge}, consistent={((x_overlap and y_overlap) == has_edge)}")
        return m
    else:
        print("No solution found")
        return None

if __name__ == "__main__":
    import re
    import os
    from z3 import Int  # keep this if you use Z3 Int(...) later

    def parse_filename(filename):
        """Parse a, b, c from filename like 'cycle_4_6_5.adjlist' or 'cycle_4_6_5_cfg0.adjlist'"""
        match = re.match(r'cycle_(\d+)_(\d+)_(\d+)(?:_cfg\d+)?\.adjlist$', filename)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
        return None

    def load_adjacency_list(file_path):
        """Load an adjacency list from a file (expects a Python literal)"""
        with open(file_path, 'r') as f:
            content = f.read().strip()
            return eval(content)

    input_dir = "test_dir"  # change as needed
    output_dir = "solved_configurations"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".adjlist"):
            continue

        params = parse_filename(filename)
        if params is None:
            print(f"Skipping file with unexpected format: {filename}")
            a=b=c=11
            # continue
        else:
            a, b, c = params
        file_path = os.path.join(input_dir, filename)

        # preserve the full base name (keeps _cfgN if present)
        base_name = os.path.splitext(filename)[0]

        print(f"\n=== Processing: {filename} (a={a}, b={b}, c={c}) ===")
        try:
            adj_list = load_adjacency_list(file_path)
            n = len(adj_list)

            solution = solve_rectangle_problem(n, adj_list)

            if solution:
                # LPR gap (float)
                lpr_gap = (a + b) / b

                # use the original base_name so any _cfgN is preserved
                output_filename = f"gap_{lpr_gap:.3f}_{base_name}.config"
                output_path = os.path.join(output_dir, output_filename)

                # reconstruct variable objects to read values from the model
                x1_vars = [Int(f"x1_{i}") for i in range(n)]
                x2_vars = [Int(f"x2_{i}") for i in range(n)]
                y1_vars = [Int(f"y1_{i}") for i in range(n)]
                y2_vars = [Int(f"y2_{i}") for i in range(n)]

                with open(output_path, 'w') as f:
                    f.write(f"# Solution for {base_name}\n")
                    f.write(f"# LPR gap: {lpr_gap:.3f}\n")
                    f.write(f"# Parameters: a={a}, b={b}, c={c}\n")
                    f.write("# Rectangle coordinates:\n")
                    f.write("# node x1 x2 y1 y2\n")
                    for i in range(n):
                        xi1 = solution[x1_vars[i]].as_long()
                        xi2 = solution[x2_vars[i]].as_long()
                        yi1 = solution[y1_vars[i]].as_long()
                        yi2 = solution[y2_vars[i]].as_long()
                        f.write(f"{i} {xi1} {xi2} {yi1} {yi2}\n")

                print(f"✓ Solution written to: {output_filename}")
            else:
                print(f"✗ No solution found for {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\nProcessing complete. Check {output_dir} for solution files.")



# if __name__ == "__main__":
#     def parse_filename(filename):
#         """Parse a, b, c from filename like 'cycle_4_6_5.adjlist'"""
#         match = re.match(r'cycle_(\d+)_(\d+)_(\d+)\.adjlist', filename)
#         if match:
#             return int(match.group(1)), int(match.group(2)), int(match.group(3))
#         return None

#     def load_adjacency_list(file_path):
#         """Load an adjacency list from a file (expects a Python literal)"""
#         with open(file_path, 'r') as f:
#             content = f.read().strip()
#             return eval(content)

#     input_dir = "out_connected_cycles"  # change as needed
#     output_dir = "solved_configurations"
#     os.makedirs(output_dir, exist_ok=True)

#     for filename in os.listdir(input_dir):
#         if not filename.endswith(".adjlist"):
#             continue

#         params = parse_filename(filename)
#         if params is None:
#             print(f"Skipping file with unexpected format: {filename}")
#             continue
#         a, b, c = params
#         file_path = os.path.join(input_dir, filename)

#         print(f"\n=== Processing: {filename} (a={a}, b={b}, c={c}) ===")
#         try:
#             adj_list = load_adjacency_list(file_path)
#             n = len(adj_list)

#             solution = solve_rectangle_problem(n, adj_list)

#             if solution:
#                 # LPR gap (float)
#                 lpr_gap = (a + b) / b

#                 output_filename = f"gap_{lpr_gap:.3f}_cycle_{a}_{b}_{c}.config"
#                 output_path = os.path.join(output_dir, output_filename)

#                 # reconstruct variable objects to read values from the model
#                 x1_vars = [Int(f"x1_{i}") for i in range(n)]
#                 x2_vars = [Int(f"x2_{i}") for i in range(n)]
#                 y1_vars = [Int(f"y1_{i}") for i in range(n)]
#                 y2_vars = [Int(f"y2_{i}") for i in range(n)]

#                 with open(output_path, 'w') as f:
#                     f.write(f"# Solution for cycle_{a}_{b}_{c}\n")
#                     f.write(f"# LPR gap: {lpr_gap:.3f}\n")
#                     f.write(f"# Parameters: a={a}, b={b}, c={c}\n")
#                     f.write("# Rectangle coordinates:\n")
#                     f.write("# node x1 x2 y1 y2\n")
#                     for i in range(n):
#                         xi1 = solution[x1_vars[i]].as_long()
#                         xi2 = solution[x2_vars[i]].as_long()
#                         yi1 = solution[y1_vars[i]].as_long()
#                         yi2 = solution[y2_vars[i]].as_long()
#                         f.write(f"{i} {xi1} {xi2} {yi1} {yi2}\n")

#                 print(f"✓ Solution written to: {output_filename}")
#             else:
#                 print(f"✗ No solution found for {filename}")

#         except Exception as e:
#             print(f"Error processing {filename}: {e}")

#     print(f"\nProcessing complete. Check {output_dir} for solution files.")

