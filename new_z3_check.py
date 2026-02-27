#!/usr/bin/env python3
# z3_rectangle_solver.py

import os
import re
import argparse
import networkx as nx
from z3 import Solver, Int, Real, Optimize, And, Or, Not, sat

def solve_mis_ilp_and_lpr(n, adjacency_list):
    """
    Solves ILP using pairwise constraints, and LPR using maximal clique constraints.
    Returns (ilp_val, ilp_nodes, lpr_val, lpr_nodes).
    """
    # --- 1. ILP (Integer) - Pairwise Constraints Only ---
    opt_ilp = Optimize()
    x_ilp = [Int(f"x_ilp_{i}") for i in range(n)]
    
    for i in range(n):
        opt_ilp.add(x_ilp[i] >= 0, x_ilp[i] <= 1)
        
    for i in range(n):
        for j in adjacency_list.get(i, []):
            if i < j:
                opt_ilp.add(x_ilp[i] + x_ilp[j] <= 1)
                
    opt_ilp.maximize(sum(x_ilp))
    opt_ilp.check()
    m_ilp = opt_ilp.model()
    
    ilp_nodes = []
    ilp_val = 0.0
    for i in range(n):
        val = m_ilp[x_ilp[i]]
        if val is not None and val.as_long() == 1:
            ilp_nodes.append(i)
            ilp_val += 1.0

    # --- 2. LPR (Real/Fractional) - Maximal Clique Constraints ---
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, neighbors in adjacency_list.items():
        for j in neighbors:
            G.add_edge(i, j)
            
    maximal_cliques = list(nx.find_cliques(G))

    opt_lpr = Optimize()
    x_lpr = [Real(f"x_lpr_{i}") for i in range(n)]
    
    for i in range(n):
        opt_lpr.add(x_lpr[i] >= 0, x_lpr[i] <= 1)
        
    for clique in maximal_cliques:
        opt_lpr.add(sum(x_lpr[i] for i in clique) <= 1)
                
    opt_lpr.maximize(sum(x_lpr))
    opt_lpr.check()
    m_lpr = opt_lpr.model()
    
    lpr_nodes = []
    lpr_val = 0.0
    for i in range(n):
        val = m_lpr[x_lpr[i]]
        if val is None:
            f_val = 0.0
        else:
            if val.is_int():
                f_val = float(val.as_long())
            else:
                f_val = float(val.numerator_as_long()) / float(val.denominator_as_long())
        
        lpr_nodes.append((i, f_val))
        lpr_val += f_val

    return ilp_val, ilp_nodes, lpr_val, lpr_nodes


def solve_rectangle_problem(n, adjacency_list):
    """Solve and display rectangle intersection constraints using Z3."""
    s = Solver()
    s.set("timeout", 30000)

    grid_size = 2 * n
    x1 = [Int(f"x1_{i}") for i in range(n)]
    x2 = [Int(f"x2_{i}") for i in range(n)]
    y1 = [Int(f"y1_{i}") for i in range(n)]
    y2 = [Int(f"y2_{i}") for i in range(n)]

    for i in range(n):
        s.add(x1[i] >= 0,        x1[i] <= grid_size - 2)
        s.add(x2[i] >= 1,        x2[i] <= grid_size - 1)
        s.add(y1[i] >= 0,        y1[i] <= grid_size - 2)
        s.add(y2[i] >= 1,        y2[i] <= grid_size - 1)
        s.add(x1[i] <= x2[i] - 1)
        s.add(y1[i] <= y2[i] - 1)

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

            if j in adjacency_list.get(i, []) or i in adjacency_list.get(j, []):
                s.add(x_overlap)
                s.add(y_overlap)
            else:
                s.add(Or(Not(x_overlap), Not(y_overlap)))

    print(f"Solving for {n} rectangles...")
    if s.check() == sat:
        return s.model()
    return None

def parse_filename(filename):
    match = re.match(r'cycle_(\d+)_(\d+)_(\d+)(?:_cfg\d+)?\.adjlist$', filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None

def load_adjacency_list(file_path):
    with open(file_path, 'r') as f:
        content = f.read().strip()
        return eval(content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="bi_jewelled_output_a8")
    parser.add_argument("--output-dir", default="solved_configurations")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for filename in os.listdir(args.input_dir):
        if not filename.endswith(".adjlist"):
            continue

        file_path = os.path.join(args.input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        params = parse_filename(filename)
        
        a, b, c = params if params else ("N/A", "N/A", "N/A")

        print(f"\n=== Processing: {filename} ===")
        try:
            adj_list = load_adjacency_list(file_path)
            n = len(adj_list)

            ilp_val, ilp_nodes, lpr_val, lpr_nodes = solve_mis_ilp_and_lpr(n, adj_list)
            
            lpr_gap = lpr_val / ilp_val if ilp_val > 0 else 0.0
            print(f"ILP Value: {ilp_val}, LPR Value: {lpr_val:.3f}, Calculated Gap: {lpr_gap:.3f}")

            solution = solve_rectangle_problem(n, adj_list)

            if solution:
                output_base = f"gap_{lpr_gap:.3f}_{base_name}"
                config_path = os.path.join(args.output_dir, f"{output_base}.config")
                solution_path = os.path.join(args.output_dir, f"{output_base}.solution")

                if not args.dry_run:
                    x1_vars = [Int(f"x1_{i}") for i in range(n)]
                    x2_vars = [Int(f"x2_{i}") for i in range(n)]
                    y1_vars = [Int(f"y1_{i}") for i in range(n)]
                    y2_vars = [Int(f"y2_{i}") for i in range(n)]

                    with open(config_path, 'w') as f:
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

                    with open(solution_path, 'w') as f:
                        f.write(f"ILP nodes: {ilp_nodes}\n")
                        lpr_formatted = [(node, round(val, 3)) for node, val in lpr_nodes]
                        f.write(f"LPR nodes: {lpr_formatted}\n")

                    print(f"✓ Saved configs to: {config_path}")
                    print(f"✓ Saved ILP/LPR to: {solution_path}")
                else:
                    print(f"✓ [DRY RUN] Would have saved solutions for {base_name}")
            else:
                print(f"✗ No rectangle intersection solution found for {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\nProcessing complete.")
