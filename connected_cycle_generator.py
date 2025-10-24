#!/usr/bin/env python3
"""
connected_cycles_generator.py

Generate connected cycles graphs and output adjacency lists for the rectangle solver.
Each call to generate_connected_cycle will produce *one file per valid cross-connection configuration*
named: {outdir}/{base_filename}_cfg{idx}.adjlist
"""

import os
from pathlib import Path
from typing import Iterable, Tuple, List, Dict, Optional

def generate_connected_cycle(a: int, b: int, c: int,
                             outdir: str = "output_connected_cycles",
                             filename: Optional[str] = None) -> List[str]:
    """
    Generate graphs for *all* cross-connection configurations satisfying:
      - each inner node connects to one even-position outer node and one odd-position outer node
      - no triangles are created by those cross-connections (local constraints enforced)

    Returns a list of produced file paths (strings).
    The 'c' parameter is only used for the base filename (so different c -> different base name).
    """
    # Basic validation
    if not (isinstance(a, int) and a >= 1):
        raise ValueError("a must be a positive integer")
    if not (isinstance(b, int) and b >= 1):
        raise ValueError("b must be a positive integer")
    if not isinstance(c, int):
        raise ValueError("c must be an integer")

    os.makedirs(outdir, exist_ok=True)
    if filename is None:
        filename = f"cycle_{a}_{b}_{c}.adjlist"
    base_outpath = Path(outdir) / filename

    total_nodes = a + b  # nodes 0..a-1 inner, a..a+b-1 outer

    def _pos_adjacent(p: int, q: int, b_len: int) -> bool:
        d = (p - q) % b_len
        return d == 1 or d == b_len - 1

    # even/odd outer position pools (positions 0..b-1)
    even_pos = [p for p in range(b) if (p % 2) == 0]
    odd_pos  = [p for p in range(b) if (p % 2) == 1]
    if not even_pos or not odd_pos:
        raise ValueError("outer cycle has no even or no odd positions (b too small)")

    # allowed (even, odd) pairs for a single inner node that are not adjacent on outer cycle
    allowed_pairs = [(e, o) for e in even_pos for o in odd_pos if not _pos_adjacent(e, o, b)]

    # backtracking to assign one (e,o) pair per inner node i, with adjacent-inner disjointness
    configs: List[List[Tuple[int,int]]] = []
    current: List[Optional[Tuple[int,int]]] = [None] * a

    def backtrack(i: int):
        if i == a:
            # full assignment: ensure wrap-around adjacency constraint (last vs first)
            if set(current[0]).isdisjoint(set(current[a-1])):
                configs.append([pair for pair in current])  # type: ignore[arg-type]
            return

        for pair in allowed_pairs:
            pair_set = set(pair)
            # conflict with previous inner node?
            if i > 0:
                prev = current[i-1]
                if prev is not None and not pair_set.isdisjoint(set(prev)):
                    continue
            # early wrap check if we're at last and first is already assigned
            if i == a - 1 and current[0] is not None:
                if not pair_set.isdisjoint(set(current[0])):
                    continue
            current[i] = pair
            backtrack(i + 1)
            current[i] = None

    backtrack(0)

    if not configs:
        raise RuntimeError("No valid cross-connection configurations found under the constraints.")

    produced_files: List[str] = []
    base_dir = base_outpath.parent
    base_name = base_outpath.stem
    base_dir.mkdir(parents=True, exist_ok=True)

    # helper to add edges for a single config's adjacency list
    def _build_and_write_cfg(cfg: List[Tuple[int,int]], idx: int):
        adj_list_cfg: Dict[int, List[int]] = {node: [] for node in range(total_nodes)}

        def _add_edge(u: int, v: int) -> None:
            if u == v:
                return
            if v not in adj_list_cfg[u]:
                adj_list_cfg[u].append(v)
            if u not in adj_list_cfg[v]:
                adj_list_cfg[v].append(u)

        # inner cycle edges
        for i in range(a - 1):
            _add_edge(i, i + 1)
        _add_edge(a - 1, 0)

        # outer cycle edges (positions 0..b-1 map to nodes a+pos)
        for pos in range(b - 1):
            u = a + pos
            v = a + (pos + 1)
            _add_edge(u, v)
        _add_edge(a + b - 1, a)

        # cross connections for this config
        for i in range(a):
            e_pos, o_pos = cfg[i]
            _add_edge(i, a + e_pos)
            _add_edge(i, a + o_pos)

        # sort and write
        for node in adj_list_cfg:
            adj_list_cfg[node].sort()

        outfile = base_dir / f"{base_name}_cfg{idx}.adjlist"
        with open(outfile, "w") as f:
            f.write(str(adj_list_cfg))
        produced_files.append(str(outfile))

    for idx, cfg in enumerate(configs):
        _build_and_write_cfg(cfg, idx)

    return produced_files


def generate_many(params: Iterable[Tuple[int, int, int]], outdir: str = "connected_cycles") -> List[str]:
    """Generate many graphs. Each item in `params` must be a 3-tuple (a,b,c).
    Returns a flattened list of all written file paths.
    """
    written: List[str] = []
    for a, b, c in params:
        paths = generate_connected_cycle(a, b, c, outdir=outdir)
        written.extend(paths)
    return written


def load_adjacency_list(file_path: str) -> Dict[int, List[int]]:
    with open(file_path, 'r') as f:
        content = f.read().strip()
        return eval(content)


if __name__ == "__main__":
    outdir = "out_connected_cycles"
    os.makedirs(outdir, exist_ok=True)

    params: List[Tuple[int,int,int]] = []
    # example small parameter generation (I left your original ranges but shortened for example)
    for a in range(6, 7):
        if a % 2 != 0:
            continue
        for b in range(12, 13):
            if b % 2 != 0:
                continue
            if b < a:
                continue
            proj_ratio = ((a+b)/2)/(b/2)
            if proj_ratio < 1.4:
                continue

            # your c-generation code (keeps the same behaviour)
            raw_target = [r for r in range(b) if (r % 2 == 1 and r > 2 and r != b - 1)]
            target_residues = { (r if r <= (b - r) else (b - r)) for r in raw_target }
            if not target_residues:
                continue

            seen = set()
            c = 3
            max_steps = b * b
            steps = 0
            while seen != target_residues and steps < max_steps:
                r = c % b
                if (r % 2 == 1) and (r > 2) and (r != b - 1):
                    canonical = r if r <= (b // 2) else (b - r)
                    if canonical in target_residues and canonical not in seen:
                        params.append((a, b, c))
                        seen.add(canonical)
                c += 2
                steps += 1

    outfiles = generate_many(params, outdir=outdir)
    print(f"Generated {len(outfiles)} adjacency list files in '{outdir}/'")


# """
# connected_cycles_generator.py

# Generate connected cycles graphs and output adjacency lists for the rectangle solver.
# """

# import os
# from typing import Iterable, Tuple, List, Dict


# def generate_connected_cycle(a: int, b: int, c: int, outdir: str = "output_connected_cycles", filename: str = None) -> str:
#     """Generate one connected-cycles graph and write its adjacency list to a file.

#     Conventions:
#     - Inner cycle nodes: 0 .. a-1
#     - Outer cycle nodes: a .. a+b-1
#     - For inner node i (0-based) connect to outer nodes at indices:
#         a + (i % b) and a + ((i + c) % b)

#     Parameters
#     ----------
#     a, b, c : int
#         Sizes for inner (a) and outer (b) cycles, and the connection offset c.
#     outdir : str
#         Output directory where the .adjlist file will be written.
#     filename : str, optional
#         Custom filename. If None a default name `cycle_{a}_{b}_{c}.adjlist` is used.

#     Returns
#     -------
#     str
#         Path to the written file.
#     """

#     # Basic validation
#     if not (isinstance(a, int) and a >= 1):
#         raise ValueError("a must be a positive integer")
#     if not (isinstance(b, int) and b >= 1):
#         raise ValueError("b must be a positive integer")
#     if not isinstance(c, int):
#         raise ValueError("c must be an integer")

#     os.makedirs(outdir, exist_ok=True)

#     if filename is None:
#         filename = f"cycle_{a}_{b}_{c}.adjlist"

#     outpath = os.path.join(outdir, filename)

#     # Build adjacency list as a dictionary
#     adj_list: Dict[int, List[int]] = {}
    
#     # Initialize empty lists for all nodes
#     total_nodes = a + b
#     for node in range(total_nodes):
#         adj_list[node] = []

#     def add_edge(u: int, v: int) -> None:
#         if u == v:
#             return
#         if v not in adj_list[u]:
#             adj_list[u].append(v)
#         if u not in adj_list[v]:
#             adj_list[v].append(u)

#     # Notive we do modulo (a-1) and (b-1) because we are zero indexed
#     # inner cycle: nodes 0..a-1
#     for i in range(a-1):
#         j = i + 1
#         add_edge(i, j)
#     add_edge(a-1, 0)  # close the inner cycle

#     # outer cycle: nodes a..a+b-1, positions 0..b-1 map to a+pos
#     for pos in range(b-1):
#         u = a + pos
#         v = a + (pos + 1)
#         add_edge(u, v)
#     add_edge(a + b - 1, a)  # close the outer cycle

#             # # cross-connections: for each inner node i connect to two outer nodes
#             # for i in range(a):
#             #     outer1 = a + (i % b)
#             #     outer2 = a + ((i + c) % b)
#             #     add_edge(i, outer1)
#             #     add_edge(i, outer2)

#     # --- REPLACEMENT: enumerate cross-connection configurations (no 'c' parameter) ---
#     # assumptions:
#     #   inner nodes: 0..a-1
#     #   outer nodes: a..a+b-1  (positions 0..b-1 map to a+pos)
#     # Writes files named: {outpath_base}_cfg{idx}.txt

#     def _pos_adjacent(p, q, b):
#         """True if positions p and q are adjacent on outer cycle of length b."""
#         d = (p - q) % b
#         return d == 1 or d == b - 1

#     # prepare even/odd position pools (positions are 0..b-1)
#     even_pos = [p for p in range(b) if (p % 2) == 0]
#     odd_pos  = [p for p in range(b) if (p % 2) == 1]
#     if not even_pos or not odd_pos:
#         # if there are no even or no odd positions, there are no valid configs
#         raise ValueError("outer cycle has no even or no odd positions (b too small)")

#     # precompute allowed per-node choices: pairs (even_pos, odd_pos) that are not adjacent
#     allowed_pairs = [(e, o) for e in even_pos for o in odd_pos if not _pos_adjacent(e, o, b)]

#     # backtracking: assign one (e,o) pair per inner node, ensuring adjacent-inner disjointness
#     configs = []
#     current = [None] * a  # current[i] = (e_pos, o_pos)

#     def backtrack(i):
#         # assign for inner node i
#         if i == a:
#             # full assignment: check wrap-around constraint between last and first
#             last = set(current[a-1])
#             first = set(current[0])
#             if last.isdisjoint(first):
#                 configs.append(list(current))  # store a copy
#             return

#         for pair in allowed_pairs:
#             pair_set = set(pair)
#             # check with previous node (i-1) if exists: they must be disjoint
#             if i > 0:
#                 prev_set = set(current[i-1])
#                 if not pair_set.isdisjoint(prev_set):
#                     continue
#             # fast prune: if i == a-1 also check disjoint with current[0] (wrap) to reduce work
#             if i == a-1 and current[0] is not None:
#                 if not pair_set.isdisjoint(set(current[0])):
#                     continue

#             current[i] = pair
#             backtrack(i + 1)
#             current[i] = None

#     # Run backtracking (be aware: this can be exponential for large a/b)
#     backtrack(0)

#     # Now write each valid configuration as its own adjacency file
#     # base_outpath: use same outpath but append _cfg{idx}
#     base_outpath = Path(outpath)
#     base_dir = base_outpath.parent
#     base_name = base_outpath.stem
#     base_dir.mkdir(parents=True, exist_ok=True)

#     if not configs:
#         raise RuntimeError("No valid cross-connection configurations found under the constraints.")

#     for idx, cfg in enumerate(configs):
#         # rebuild adjacency list from scratch for each configuration
#         adj_list_cfg: Dict[int, List[int]] = {node: [] for node in range(total_nodes)}

#         # inner cycle
#         for i in range(a - 1):
#             add_edge = lambda u, v: (adj_list_cfg[u].append(v) if v not in adj_list_cfg[u] else None,
#                                     adj_list_cfg[v].append(u) if u not in adj_list_cfg[v] else None)
#         # we already have an add_edge helper earlier in your code; reuse it by copying references:
#         # But to keep things self-contained here, use a small local helper:
#         def _add_edge(u: int, v: int) -> None:
#             if u == v:
#                 return
#             if v not in adj_list_cfg[u]:
#                 adj_list_cfg[u].append(v)
#             if u not in adj_list_cfg[v]:
#                 adj_list_cfg[v].append(u)

#         # inner cycle edges
#         for i in range(a - 1):
#             _add_edge(i, i + 1)
#         _add_edge(a - 1, 0)

#         # outer cycle edges (positions 0..b-1 map to nodes a+pos)
#         for pos in range(b - 1):
#             u = a + pos
#             v = a + (pos + 1)
#             _add_edge(u, v)
#         _add_edge(a + b - 1, a)

#         # cross-connections from current cfg
#         for i in range(a):
#             e_pos, o_pos = cfg[i]
#             outer1 = a + e_pos
#             outer2 = a + o_pos
#             _add_edge(i, outer1)
#             _add_edge(i, outer2)

#         # sort neighbors for deterministic output
#         for node in adj_list_cfg:
#             adj_list_cfg[node].sort()

#         outfile = base_dir / f"{base_name}_cfg{idx}.txt"
#         with open(outfile, "w") as f:
#             f.write(str(adj_list_cfg))

#     # end replacement block


#     # Sort neighbor lists for consistent output
#     for node in adj_list:
#         adj_list[node].sort()

#     # Write adjacency list in format expected by the solver
#     with open(outpath, "w") as f:
#         # Write as a Python dictionary that can be eval'd
#         f.write(str(adj_list))

#     return outpath


# def generate_many(params: Iterable[Tuple[int, int, int]], outdir: str = "connected_cycles") -> List[str]:
#     """Generate many graphs. Each item in `params` must be a 3-tuple (a,b,c).

#     Returns a list of written file paths.
#     """
#     written = []
#     for a, b, c in params:
#         path = generate_connected_cycle(a, b, c, outdir=outdir)
#         written.append(path)
#     return written


# def load_adjacency_list(file_path: str) -> Dict[int, List[int]]:
#     """Load an adjacency list from a file created by this generator."""
#     with open(file_path, 'r') as f:
#         content = f.read().strip()
#         # Safely evaluate the string as a Python dictionary
#         return eval(content)


# if __name__ == "__main__":
#     outdir = "out_connected_cycles"
#     os.makedirs(outdir, exist_ok=True)

#     params = []
#     # a in range(4,21) even only
#     for a in range(4, 9):
#         if a % 2 != 0:
#             continue
#         # b in range(6,41) and b <= 2a for ILP ratio
#         for b in range(6, 17):
#             if b<a:
#                 continue
#             #if b > 2 * a:
#                 #continue
#             proj_ratio = ((a+b)/2)/(b/2)
#             if proj_ratio < 1.4:
#                 continue
            
#             # --- new c-generation (allow c=3, avoid ± duplicates, skip b-1) ---
#             # Build raw target residues: odd residues r with r > 2 and r != b-1
#             raw_target = [r for r in range(b) if (r % 2 == 1 and r > 2 and r != b - 1)]

#             # Collapse each pair {r, b-r} to a canonical representative (the smaller of the pair)
#             target_residues = { (r if r <= (b - r) else (b - r)) for r in raw_target }

#             if not target_residues:
#                 continue

#             seen = set()
#             c = 3               # start at 3 (include 3)
#             max_steps = b * b   # safety cap
#             steps = 0

#             while seen != target_residues and steps < max_steps:
#                 r = c % b
#                 # Skip b-1 residues explicitly
#                 if (r % 2 == 1) and (r > 2) and (r != b - 1):
#                     canonical = r if r <= (b // 2) else (b - r)
#                     if canonical in target_residues and canonical not in seen:
#                         params.append((a, b, c))
#                         seen.add(canonical)
#                 c += 2  # next odd
#                 steps += 1

#             # optional: warning if not all target residues covered
#             if seen != target_residues:
#                 # print(f"Warning: for b={b} covered {seen} of {target_residues}")
#                 pass
#             # --- end c-generation ---


#     # Generate and save
#     outfiles = generate_many(params, outdir=outdir)
#     print(f"Generated {len(outfiles)} adjacency list files in '{outdir}/'")
