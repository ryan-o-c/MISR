#!/usr/bin/env python3
"""
connected_cycles_generator.py

Generate connected cycles graphs and output adjacency lists for the rectangle solver.
Each call to generate_connected_cycle will produce *one file per valid cross-connection configuration*
named: {outdir}/{base_filename}_cfg{idx}.adjlist
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Iterable
from z3 import Bool, And, Not, Optimize, If, Sum, is_true
import json
from pathlib import Path

def mis_z3(adj_list: Dict[int, List[int]],
           time_limit_ms: Optional[int] = None
           ) -> Tuple[int, Set[int]]:
    """
    Solve Maximum Independent Set using Z3 Optimize.

    Args:
        adj_list: adjacency list mapping node id -> list of neighbor node ids (undirected).
        time_limit_ms: optional solver time limit in milliseconds.

    Returns:
        (mis_size, mis_nodes) where mis_nodes is a set of node ids from adj_list.
    """
    # Sort nodes to have a deterministic order
    nodes = sorted(adj_list.keys())

    # Create a Bool variable for each node (True if selected in the independent set)
    zvars = {node: Bool(f"x_{node}") for node in nodes}

    opt = Optimize()

    # Optional time limit
    if time_limit_ms is not None:
        opt.set("timeout", time_limit_ms)

    # Add constraints: for every undirected edge (u,v) we must not select both u and v.
    # To avoid duplicate constraints, only add for ordered pairs (u < v)
    for u in nodes:
        for v in adj_list[u]:
            # only add once, when u comes before v in our ordering
            if u < v:
                opt.add(Not(And(zvars[u], zvars[v])))

    # Objective: maximize sum of selected variables
    # Convert booleans to integers via If(var, 1, 0)
    sum_selected = Sum([If(zvars[node], 1, 0) for node in nodes])
    opt.maximize(sum_selected)

    # Solve
    res = opt.check()
    if res.r is None:  # sometimes Optimize returns different status objects; handle conservatively
        # No model found
        return 0, set()

    model = opt.model()

    # Extract selected nodes
    selected = {node for node in nodes if is_true(model.evaluate(zvars[node]))}

    return len(selected), selected

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def generate_connected_cycle(a: int, b: int, c: int,
                             outdir: str = "output_connected_cycles",
                             filename: Optional[str] = None,
                             show_progress: bool = True,
                             print_every: int = 25) -> List[str]:
    """
    Generate graphs for all cross-connection configurations satisfying:
      - each inner node connects to one even-position outer node and one odd-position outer node
      - no triangles are created by those cross-connections (local constraints enforced)

    Writes adjacency lists exactly as before to files named:
      <outdir>/<filename_stem>_cfg{idx}.adjlist
    (content = str(adj_list_dict)).

    Progress printing:
      - show_progress: if True, prints progress. Uses tqdm if available.
      - print_every: frequency for full status lines when tqdm is not installed.

    Returns list of written file paths (strings).
    """
    # --- Validate inputs ---
    if not (isinstance(a, int) and a >= 1):
        raise ValueError("a must be a positive integer")
    if not (isinstance(b, int) and b >= 1):
        raise ValueError("b must be a positive integer")
    if not isinstance(c, int):
        raise ValueError("c must be an integer")

    # --- Prepare output path & filename ---
    os.makedirs(outdir, exist_ok=True)
    if filename is None:
        filename = f"cycle_{a}_{b}_{c}.adjlist"
    base_path = Path(outdir) / filename
    base_dir = base_path.parent
    base_name = base_path.stem
    base_dir.mkdir(parents=True, exist_ok=True)

    total_nodes = a + b  # nodes 0..a-1 inner, a..a+b-1 outer

    def _pos_adjacent(p: int, q: int, b_len: int) -> bool:
        d = (p - q) % b_len
        return d == 1 or d == (b_len - 1)

    # build pools of even and odd outer positions (0..b-1)
    even_positions = [p for p in range(b) if (p % 2) == 0]
    odd_positions = [p for p in range(b) if (p % 2) == 1]
    if not even_positions or not odd_positions:
        raise ValueError("outer cycle has no even or no odd positions (b too small)")

    # allowed pairs (even_pos, odd_pos) that are not adjacent on outer cycle
    allowed_pairs: List[Tuple[int, int]] = [
        (e, o) for e in even_positions for o in odd_positions
        if not _pos_adjacent(e, o, b)
    ]
    if not allowed_pairs:
        raise RuntimeError("No allowed (even, odd) cross-connection pairs available.")

    # --- Backtracking: assign one allowed (e,o) to each inner node with adjacency disjointness ---
    configs: List[List[Tuple[int, int]]] = []
    current: List[Optional[Tuple[int, int]]] = [None] * a

    def backtrack_assign(i: int) -> None:
        if i == a:
            # full assignment: ensure first and last are disjoint (wrap-around)
            first = current[0]
            last = current[-1]
            assert first is not None and last is not None
            if set(first).isdisjoint(set(last)):
                configs.append([pair for pair in current])  # copy
            return

        for pair in allowed_pairs:
            # check disjointness with previous inner node (if any)
            if i > 0:
                prev = current[i - 1]
                if prev is not None and not set(pair).isdisjoint(set(prev)):
                    continue
            # if placing last, also check disjointness with first (if first already set)
            if i == a - 1 and current[0] is not None:
                if not set(pair).isdisjoint(set(current[0])):
                    continue

            current[i] = pair
            backtrack_assign(i + 1)
            current[i] = None

    backtrack_assign(0)

    if not configs:
        raise RuntimeError("No valid cross-connection configurations found under the constraints.")

    produced_files: List[str] = []

    def _build_adjlist_for_config(cfg: List[Tuple[int, int]]) -> Dict[int, List[int]]:
        adj: Dict[int, List[int]] = {node: [] for node in range(total_nodes)}

        def add_edge(u: int, v: int) -> None:
            if u == v:
                return
            if v not in adj[u]:
                adj[u].append(v)
            if u not in adj[v]:
                adj[v].append(u)

        # inner cycle edges 0..a-1
        for i in range(a):
            add_edge(i, (i + 1) % a)

        # outer cycle edges a..a+b-1 (positions map to a + pos)
        for pos in range(b):
            u = a + pos
            v = a + ((pos + 1) % b)
            add_edge(u, v)

        # cross-connections: inner i connects to a + e_pos and a + o_pos
        for i in range(a):
            e_pos, o_pos = cfg[i]
            add_edge(i, a + e_pos)
            add_edge(i, a + o_pos)

        # sort neighbor lists for determinism
        for node in adj:
            adj[node].sort()

        return adj

    # --- Progress helpers ---
    use_tqdm = False
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore
            use_tqdm = True
        except Exception:
            use_tqdm = False

    total_configs = len(configs)
    start_time = time.time()
    processed = 0
    written = 0
    skipped = 0

    # iterate over configs with progress printing; use indices explicitly so filenames match original idx
    indices = list(range(total_configs))
    if use_tqdm:
        from tqdm import tqdm  # type: ignore
        indices = list(tqdm(indices, desc="configs", unit="cfg"))

    for idx in indices:
        cfg = configs[idx]
        proc_start = time.time()
        adj_list_cfg = _build_adjlist_for_config(cfg)

        # compute MIS using the external mis_z3 function (must be present)
        try:
            mis_size, mis_nodes = mis_z3(adj_list_cfg)  # type: ignore[name-defined]
        except NameError as exc:
            raise RuntimeError(
                "mis_z3 is not defined in this module. Provide mis_z3(adj_list) as shown earlier."
            ) from exc

        processed += 1
        n = total_nodes
        if mis_size > n/3:
            skipped += 1
        elif mis_size==0:
            print("Error solving here!")
            skipped+=1
        else:
            # WRITE using the original style requested by you
            outfile = base_dir / f"{base_name}_cfg{idx}.adjlist"
            with open(outfile, "w") as f:
                f.write(str(adj_list_cfg))
            produced_files.append(str(outfile))
            written += 1

        # progress bookkeeping & printing
        proc_end = time.time()
        elapsed = proc_end - start_time
        avg_per = elapsed / processed if processed else 0.0
        remaining = max(0, total_configs - processed)
        est_remain = remaining * avg_per

        if show_progress and not use_tqdm:
            percent = (processed / total_configs) * 100
            compact = (
                f"\rProcessed: {processed}/{total_configs} "
                f"({percent:.1f}%) | written: {written} | skipped: {skipped} | "
                f"elapsed: {elapsed:.1f}s | avg/config: {avg_per:.3f}s"
            )
            print(compact, end="", flush=True)

            if (processed % print_every) == 0 or processed == total_configs:
                full = (
                    f"\n[Status] processed={processed}/{total_configs}, written={written}, "
                    f"skipped={skipped}, elapsed={elapsed:.1f}s, avg_per={avg_per:.3f}s, "
                    f"ETA={est_remain:.1f}s\n"
                )
                print(full, end="", flush=True)

    if show_progress and not use_tqdm:
        print()  # finish the progress line

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
    for a in range(6, 9):
        if a % 2 != 0:
            continue
        for b in range(6, (a+2)):
            # if b % 2 != 0:
                # continue
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
