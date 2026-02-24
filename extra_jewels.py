import os
import time
import itertools
from pathlib import Path
from typing import List, Tuple, Dict, Iterator
from concurrent.futures import ProcessPoolExecutor
from z3 import Bool, And, Not, Optimize, If, Sum, is_true, Context

# --- Graph Logic ---

def mis_z3(adj_list: Dict[int, List[int]]) -> int:
    """Returns the size of the Maximum Independent Set using a fresh Z3 Context."""
    ctx = Context()
    nodes = sorted(adj_list.keys())
    zvars = {node: Bool(f"x_{node}", ctx=ctx) for node in nodes}
    opt = Optimize(ctx=ctx)
    opt.set("timeout", 10000) 

    for u in nodes:
        for v in adj_list[u]:
            if u < v:
                opt.add(Not(And(zvars[u], zvars[v])))
    
    opt.maximize(Sum([If(zvars[node], 1, 0) for node in nodes]))
    
    try:
        if opt.check().r == 1:
            model = opt.model()
            return sum(1 for node in nodes if is_true(model.evaluate(zvars[node])))
    except Exception:
        return 0
    return 0

def build_adj_list(a: int, shifts: Tuple[int, ...]) -> Dict[int, List[int]]:
    adj = {i: [] for i in range(2 * a)}
    def add_edge(u, v):
        if v not in adj[u]:
            adj[u].append(v)
            adj[v].append(u)

    for i in range(a):
        add_edge(i, (i + 1) % a)
        add_edge(i + a, ((i + 1) % a) + a)
        add_edge(i, i + a)
        add_edge(i, ((i + shifts[i]) % a) + a)
    return adj

# --- Augmentation Helpers ---

def get_candidate_edges(adj: Dict[int, List[int]], n: int) -> List[Tuple[int, int]]:
    """Finds all pairs (u, v) that are NOT edges and share NO common neighbors (Triangle-Free)."""
    candidates = []
    for u in range(n):
        # We only need to check v > u
        for v in range(u + 1, n):
            if v in adj[u]: 
                continue
            # If the intersection of neighbors is empty, adding (u,v) creates no triangle
            if not set(adj[u]).intersection(adj[v]):
                candidates.append((u, v))
    return candidates

def save_graph(adj: Dict[int, List[int]], a: int, shifts_str: str, size: int, extra_count: int, outdir: str):
    """Saves the adjacency list with metadata in the filename."""
    filename = f"a{a}_gap{size}_extra{extra_count}_seq_{shifts_str}.adjlist"
    filepath = Path(outdir) / filename
    # Avoid overwriting or redundant saves
    if not filepath.exists():
        with open(filepath, "w") as f:
            f.write(str({k: sorted(v) for k, v in adj.items()}))

def recursive_augment(adj: Dict[int, List[int]], n: int, candidates: List[Tuple[int, int]], 
                      start_idx: int, a: int, shifts_str: str, outdir: str, target_size: float):
    """Recursively adds edges, checking triangle and MIS constraints."""
    
    for i in range(start_idx, len(candidates)):
        u, v = candidates[i]
        
        # Check if still triangle-free (other edges added in this branch might have changed this)
        if not set(adj[u]).intersection(adj[v]):
            # Add edge
            adj[u].append(v)
            adj[v].append(u)
            
            # Check MIS
            size = mis_z3(adj)
            
            # If MIS meets threshold (using your n/3 logic), save it
            if 0 < size <= target_size:
                save_graph(adj, a, shifts_str, size, i, outdir)
            
            # Recurse: only look at candidate edges further down the list to avoid duplicates
            recursive_augment(adj, n, candidates, i + 1, a, shifts_str, outdir, target_size)
            
            # Backtrack
            adj[u].remove(v)
            adj[v].remove(u)

# --- FKM Necklace Generation ---

def generate_unique_necklaces(a: int) -> Iterator[Tuple[int, ...]]:
    alphabet = [s for s in range(3, a, 2) if s % a != 1 and s % a != a - 1]
    n, k = a, len(alphabet)
    if k == 0: return
    arr = [0] * (n + 1)
    def gen(t, p):
        if t > n:
            if n % p == 0: yield tuple(alphabet[arr[i]] for i in range(1, n + 1))
        else:
            arr[t] = arr[t - p]
            yield from gen(t + 1, p)
            for j in range(arr[t - p] + 1, k):
                arr[t] = j
                yield from gen(t + 1, t)
    yield from gen(1, 1)

# --- Main Logic ---

def worker_task(a: int, shifts: Tuple[int, ...], outdir: str) -> bool:
    adj = build_adj_list(a, shifts)
    n = 2 * a
    target_size = n / 3
    shifts_str = "_".join(map(str, shifts))
    
    # 1. Check base necklace
    size = mis_z3(adj)
    if 0 < size <= target_size:
        save_graph(adj, a, shifts_str, size, 0, outdir)
    
    # 2. Start augmentation
    candidates = get_candidate_edges(adj, n)
    recursive_augment(adj, n, candidates, 0, a, shifts_str, outdir, target_size)
    return True

def run_parallel_generation(a: int, num_workers: int = 30):
    outdir = f"jewelled_output_a{a}"
    os.makedirs(outdir, exist_ok=True)
    start_time = time.time()
    total_processed = 0
    last_milestone = 0
    
    necklace_gen = generate_unique_necklaces(a)
    batch_size = num_workers * 2 # Smaller batches because augmentation takes longer

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        while True:
            batch = []
            try:
                for _ in range(batch_size): batch.append(next(necklace_gen))
            except StopIteration: pass
            if not batch: break
            
            futures = [executor.submit(worker_task, a, seq, outdir) for seq in batch]
            for future in futures:
                future.result()
                total_processed += 1
            
            if (total_processed // 1000) > last_milestone:
                last_milestone = total_processed // 1000
                elapsed = time.time() - start_time
                print(f"[Progress] a={a} | Necklaces: {total_processed:,} | Speed: {total_processed/elapsed:.1f}/sec")

if __name__ == "__main__":
    for test_a in range(6, 20):
        run_parallel_generation(a=test_a, num_workers=30)
