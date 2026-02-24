import os
import time
import itertools
from pathlib import Path
from typing import List, Tuple, Dict, Iterator
from concurrent.futures import ProcessPoolExecutor
from z3 import Bool, And, Not, Optimize, If, Sum, is_true, Context

def mis_z3(adj_list: Dict[int, List[int]]) -> int:
    """Returns the size of the Maximum Independent Set using a fresh Z3 Context."""
    ctx = Context()
    nodes = sorted(adj_list.keys())
    
    # Variables and Optimize MUST be bound to the context
    zvars = {node: Bool(f"x_{node}", ctx=ctx) for node in nodes}
    opt = Optimize(ctx=ctx)
    
    opt.set("timeout", 100000)

    for u in nodes:
        for v in adj_list[u]:
            if u < v:
                # Operators inherit context from zvars[u] and zvars[v]
                opt.add(Not(And(zvars[u], zvars[v])))
    
    # Sum and If also inherit context from the zvars within them
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

# --- FKM Necklace Generation (Memory Efficient) ---

def generate_unique_necklaces(a: int) -> Iterator[Tuple[int, ...]]:
    """
    FKM Algorithm generates necklace representatives directly.
    Memory usage is O(a), not O(Total_Necklaces).
    """
    # Alphabet: only the allowed odd shifts
    alphabet = [s for s in range(3, a, 2) if s % a != 1 and s % a != a - 1]
    n = a
    k = len(alphabet)
    if k == 0: return

    arr = [0] * (n + 1)
    
    def gen(t, p):
        if t > n:
            if n % p == 0:
                yield tuple(alphabet[arr[i]] for i in range(1, n + 1))
        else:
            arr[t] = arr[t - p]
            yield from gen(t + 1, p)
            for j in range(arr[t - p] + 1, k):
                arr[t] = j
                yield from gen(t + 1, t)

    yield from gen(1, 1)

# --- Worker Function ---

def worker_task(a: int, shifts: Tuple[int, ...], outdir: str) -> bool:
    adj = build_adj_list(a, shifts)
    n = 2 * a
    size = mis_z3(adj)
    
    if 0 < size <= (n / 3):
        seq_str = "_".join(map(str, shifts))
        filename = f"a{a}_seq_{seq_str}.adjlist"
        filepath = Path(outdir) / filename
        with open(filepath, "w") as f:
            f.write(str({k: sorted(v) for k, v in adj.items()}))
        return True
    return False

# --- Main Runner ---

def run_parallel_generation(a: int, num_workers: int = 30):
    outdir = f"output_a{a}"
    os.makedirs(outdir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"[*] Starting Analysis for a = {a} (n = {2*a})")
    print(f"[*] Target MIS size: <= {2*a/3:.2f}")
    print(f"{'='*50}")

    start_time = time.time()
    results_found = 0
    total_processed = 0
    
    # Small batch size to keep the executor's internal queue from eating RAM
    batch_size = num_workers * 100 
    necklace_gen = generate_unique_necklaces(a)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        while True:
            # 1. Pull a batch from the memory-efficient generator
            batch = []
            try:
                for _ in range(batch_size):
                    batch.append(next(necklace_gen))
            except StopIteration:
                pass # Generator exhausted

            if not batch:
                break
            
            # 2. Submit and immediately wait for this batch (Throttling)
            # This is key: it prevents 'executor' from storing millions of futures in RAM
            futures = [executor.submit(worker_task, a, seq, outdir) for seq in batch]
            
            for future in futures:
                try:
                    if future.result():
                        results_found += 1
                except Exception as e:
                    print(f"Worker Error: {e}")
                total_processed += 1
            
            elapsed = time.time() - start_time
            avg_speed = total_processed / elapsed if elapsed > 0 else 0
            print(f"[Progress] a={a} | Processed: {total_processed} | Found: {results_found} | Speed: {avg_speed:.1f} tasks/sec")

    print(f"\n[!] Finished a={a}. Total Valid Graphs: {results_found}")
    print(f"[!] Total Time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    # Starting from 13 as per your trace
    for test_a in range(13, 20):
        run_parallel_generation(a=test_a, num_workers=30)
