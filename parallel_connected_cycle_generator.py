#!/usr/bin/env python3
"""
connected_cycles_generator_parallel.py

Generate connected cycles graphs and output adjacency lists for the rectangle solver.
Parallelizes per-configuration processing (adjlist build, MIS via Z3, file write)
after generating all valid cross-connection configurations.

Each written file uses the original naming scheme:
    {outdir}/{base_filename}_cfg{idx}.adjlist

New argument: num_workers (int) controls number of worker processes used for
processing configurations. If num_workers <= 1, processing runs serially.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Iterable
from z3 import Bool, And, Not, Optimize, If, Sum, is_true
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import concurrent.futures

# --- MIS solver using Z3 (top-level so worker processes can import it) ---
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
    nodes = sorted(adj_list.keys())
    zvars = {node: Bool(f"x_{node}") for node in nodes}

    opt = Optimize()
    if time_limit_ms is not None:
        opt.set("timeout", time_limit_ms)

    for u in nodes:
        for v in adj_list[u]:
            if u < v:
                opt.add(Not(And(zvars[u], zvars[v])))

    sum_selected = Sum([If(zvars[node], 1, 0) for node in nodes])
    opt.maximize(sum_selected)

    res = opt.check()
    # Handle weird return types defensively
    try:
        status = res.r if hasattr(res, "r") else res
    except Exception:
        status = res

    if status is None:
        return 0, set()

    model = opt.model()
    selected = {node for node in nodes if is_true(model.evaluate(zvars[node]))}
    return len(selected), selected


# --- Worker helper (module-level so picklable) ---
def _process_single_config_worker(args):
    """
    Worker function to be executed in a separate process.

    args: tuple containing (idx, cfg, a, b, base_dir_str, base_name)
    Returns: dict with keys: idx, written (bool), skipped (bool), outfile (str|None)
    """
    idx, cfg, a, b, base_dir_str, base_name, time_limit_ms = args
    try:
        total_nodes = a + b
        # Build adj list
        adj: Dict[int, List[int]] = {node: [] for node in range(total_nodes)}

        def add_edge(u: int, v: int) -> None:
            if u == v:
                return
            if v not in adj[u]:
                adj[u].append(v)
            if u not in adj[v]:
                adj[v].append(u)

        # inner cycle edges
        for i in range(a):
            add_edge(i, (i + 1) % a)

        # outer cycle edges (positions map to a + pos)
        for pos in range(b):
            u = a + pos
            v = a + ((pos + 1) % b)
            add_edge(u, v)

        # cross-connections
        for i in range(a):
            e_pos, o_pos = cfg[i]
            add_edge(i, a + e_pos)
            add_edge(i, a + o_pos)

        for node in adj:
            adj[node].sort()

        # Compute MIS
        mis_size, mis_nodes = mis_z3(adj, time_limit_ms=time_limit_ms)

        # Apply same filters as original:
        if mis_size > total_nodes / 3:
            return {"idx": idx, "written": False, "skipped": True, "outfile": None}
        if mis_size == 0:
            # keep the original behavior of printing an error in serial mode
            return {"idx": idx, "written": False, "skipped": True, "outfile": None}

        # write the file (same format as original: str(adj))
        base_dir = Path(base_dir_str)
        outfile = base_dir / f"{base_name}_cfg{idx}.adjlist"
        # Ensure directory exists (worker should create if needed)
        base_dir.mkdir(parents=True, exist_ok=True)
        with open(outfile, "w") as f:
            f.write(str(adj))

        return {"idx": idx, "written": True, "skipped": False, "outfile": str(outfile)}
    except Exception as exc:
        # On any unexpected exception, report skipped
        return {"idx": idx, "written": False, "skipped": True, "outfile": None, "error": str(exc)}


# --- Main generator (parallelized after configs generation) ---
def generate_connected_cycle(a: int, b: int, c: int,
                             outdir: str = "output_connected_cycles",
                             filename: Optional[str] = None,
                             show_progress: bool = True,
                             print_every: int = 25,
                             num_workers: int = 4,
                             mis_time_limit_ms: Optional[int] = None
                             ) -> List[str]:
    """
    Generate graphs for all cross-connection configurations satisfying:
      - each inner node connects to one even-position outer node and one odd-position outer node
      - no triangles are created by those cross-connections (local constraints enforced)

    Parallelizes the per-configuration heavy work (adjlist -> MIS -> file write) using
    ProcessPoolExecutor with `num_workers` worker processes.

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

    # --- FIX: force inner node 0 to connect to outer position 0 (i.e. node a + 0)
    # keep only allowed start-pairs for the first inner node that use even_pos == 0
    fixed_even_pos = 0
    start_allowed_pairs: List[Tuple[int, int]] = [p for p in allowed_pairs if p[0] == fixed_even_pos]
    if not start_allowed_pairs:
        raise RuntimeError("No allowed start pairs that connect inner node 0 to outer position 0.")


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

        candidate_pairs = start_allowed_pairs if i == 0 else allowed_pairs

        for pair in candidate_pairs:
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

    # --- Decide serial vs parallel execution ---
    total_configs = len(configs)
    processed = 0
    written = 0
    skipped = 0
    errors = 0

    use_tqdm = False
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore
            use_tqdm = True
        except Exception:
            use_tqdm = False

    start_time = time.time()

    if num_workers is None or num_workers <= 1:
        # Serial processing (keeps identical behavior except progress formatting)
        for idx, cfg in enumerate(configs):
            proc_start = time.time()
            result = _process_single_config_worker((idx, cfg, a, b, str(base_dir), base_name, mis_time_limit_ms))
            processed += 1
            if result.get("written"):
                written += 1
                produced_files.append(result.get("outfile"))
            elif result.get("skipped"):
                skipped += 1
            else:
                skipped += 1
                if result.get("error"):
                    errors += 1
                    print(f"\n[Error] idx={idx} error={result.get('error')}")
            proc_end = time.time()

            # progress printing for serial (no tqdm)
            if show_progress and not use_tqdm:
                elapsed = time.time() - start_time
                avg_per = elapsed / processed if processed else 0.0
                percent = (processed / total_configs) * 100
                compact = (
                    f"\rProcessed: {processed}/{total_configs} "
                    f"({percent:.1f}%) | written: {written} | skipped: {skipped} | "
                    f"elapsed: {elapsed:.1f}s | avg/config: {avg_per:.3f}s"
                )
                print(compact, end="", flush=True)

                if (processed % print_every) == 0 or processed == total_configs:
                    remaining = max(0, total_configs - processed)
                    est_remain = remaining * avg_per
                    full = (
                        f"\n[Status] processed={processed}/{total_configs}, written={written}, "
                        f"skipped={skipped}, elapsed={elapsed:.1f}s, avg_per={avg_per:.3f}s, "
                        f"ETA={est_remain:.1f}s\n"
                    )
                    print(full, end="", flush=True)

        if show_progress and not use_tqdm:
            print()  # finish the progress line

    else:
        # Parallel processing with ProcessPoolExecutor
        # Build argument list for each config; pass mis_time_limit_ms so workers can use it
        tasks_args = [
            (idx, cfg, a, b, str(base_dir), base_name, mis_time_limit_ms)
            for idx, cfg in enumerate(configs)
        ]

        # Use ProcessPoolExecutor; careful to handle KeyboardInterrupt and propagate nicely
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # submit all tasks
                future_to_idx = {executor.submit(_process_single_config_worker, arg): arg[0] for arg in tasks_args}

                if use_tqdm:
                    from tqdm import tqdm  # type: ignore
                    pbar = tqdm(total=total_configs, desc="configs", unit="cfg")
                else:
                    pbar = None

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    processed += 1
                    try:
                        result = future.result()
                    except Exception as exc:
                        skipped += 1
                        errors += 1
                        if show_progress:
                            print(f"\n[Worker error] idx={idx} exception={exc}")
                        result = {"idx": idx, "written": False, "skipped": True, "outfile": None, "error": str(exc)}

                    if result.get("written"):
                        written += 1
                        produced_files.append(result.get("outfile"))
                    elif result.get("skipped"):
                        skipped += 1

                    if result.get("error"):
                        errors += 1

                    # update progress bar or serial-style compact print
                    if pbar is not None:
                        pbar.update(1)
                    elif show_progress and not use_tqdm:
                        elapsed = time.time() - start_time
                        avg_per = elapsed / processed if processed else 0.0
                        percent = (processed / total_configs) * 100
                        compact = (
                            f"\rProcessed: {processed}/{total_configs} "
                            f"({percent:.1f}%) | written: {written} | skipped: {skipped} | "
                            f"elapsed: {elapsed:.1f}s | avg/config: {avg_per:.3f}s"
                        )
                        print(compact, end="", flush=True)

                if pbar is not None:
                    pbar.close()

        except KeyboardInterrupt:
            # Best effort: attempt shutdown of executor then re-raise
            print("\n[Interrupted] shutting down workers...")
            raise

        # final newline for neatness if not using tqdm
        if show_progress and not use_tqdm:
            print()

    # summary
    total_elapsed = time.time() - start_time
    if show_progress:
        print(f"[Done] processed={processed}, written={written}, skipped={skipped}, errors={errors}, elapsed={total_elapsed:.1f}s")

    return produced_files


def generate_many(params: Iterable[Tuple[int, int, int]], outdir: str = "connected_cycles", num_workers: int = 4) -> List[str]:
    """Generate many graphs. Each item in `params` must be a 3-tuple (a,b,c).
    Returns a flattened list of all written file paths.
    Note: num_workers is forwarded to each generate_connected_cycle call.
    """
    written: List[str] = []
    for a, b, c in params:
        paths = generate_connected_cycle(a, b, c, outdir=outdir, num_workers=num_workers)
        written.extend(paths)
    return written


def load_adjacency_list(file_path: str) -> Dict[int, List[int]]:
    with open(file_path, 'r') as f:
        content = f.read().strip()
        # Keep original behavior (user is responsible for trusted files)
        return eval(content)


if __name__ == "__main__":
    # Example usage (kept similar to your original main block)
    outdir = "out_connected_cycles"
    os.makedirs(outdir, exist_ok=True)

    params: List[Tuple[int,int,int]] = []
    for a in range(6, 9):
        if a % 2 != 0:
            continue
        for b in range(6, (a+2)):
            if b < a:
                continue
            proj_ratio = ((a+b)/2)/(b/2)
            if proj_ratio < 1.4:
                continue

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

    # choose a sensible number of workers; for testing you may want 1 (serial) or 2..4
    num_workers = 4
    mis_time_limit_ms = None  # optionally set a time limit per MIS solve in milliseconds

    outfiles = generate_many(params, outdir=outdir, num_workers=num_workers)
    print(f"Generated {len(outfiles)} adjacency list files in '{outdir}/'")
