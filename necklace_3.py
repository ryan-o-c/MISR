import os
import time
import math
from pathlib import Path
from typing import List, Tuple, Dict, Iterator
from concurrent.futures import ProcessPoolExecutor
from z3 import Bool, And, Not, Optimize, If, Sum, is_true, Context

# --- Combinatorics & Math for Progress Tracking ---

def euler_phi(n: int) -> int:
    """Euler's totient function: counts integers up to n coprime to n."""
    amount = 0
    for k in range(1, n + 1):
        if math.gcd(n, k) == 1:
            amount += 1
    return amount

def macmahon_necklaces(n: int, k: int) -> int:
    """MacMahon's formula for the number of k-ary necklaces of length n."""
    if k == 0: return 0
    total = 0
    for d in range(1, n + 1):
        if n % d == 0:
            total += euler_phi(d) * (k ** (n // d))
    return total // n

def format_eta(seconds: float) -> str:
    """Safely formats ETA into HH:MM:SS, avoiding overflows."""
    try:
        if math.isnan(seconds) or math.isinf(seconds) or seconds < 0:
            return "??:??"
        if seconds > 31536000: # More than a year
            return ">1 Year"
        
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
    except Exception:
        return "??:??"

# --- ILP Logic ---

def mis_z3(adj_list: Dict[int, List[int]]) -> int:
    ctx = Context()
    nodes = sorted(adj_list.keys())
    
    zvars = {node: Bool(f"x_{node}", ctx=ctx) for node in nodes}
    opt = Optimize(ctx=ctx)
    opt.set("timeout", 100000)

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

# --- Graph Adjacency Logic ---

def generate_even_odd_alphabet(a: int) -> List[Tuple[int, int]]:
    pairs = []
    for s1 in range(0, a, 2):
        for s2 in range(1, a, 2):
            if (s2 - s1) % a != 1 and (s1 - s2) % a != 1:
                pairs.append((s1, s2))
    return sorted(pairs)

def build_adj_list(a: int, shift_pairs: Tuple[Tuple[int, int], ...]) -> Dict[int, List[int]]:
    adj = {i: [] for i in range(2 * a)}
    def add_edge(u, v):
        if v not in adj[u]:
            adj[u].append(v)
            adj[v].append(u)

    for i in range(a):
        add_edge(i, (i + 1) % a)
        add_edge(i + a, ((i + 1) % a) + a)
        
        s1, s2 = shift_pairs[i]
        add_edge(i, ((i + s1) % a) + a)
        add_edge(i, ((i + s2) % a) + a)
        
    return adj

# --- FKM Necklace Generation ---

def generate_unique_necklaces(a: int) -> Iterator[Tuple[Tuple[int, int], ...]]:
    alphabet = generate_even_odd_alphabet(a)
    n = a
    k = len(alphabet)
    if k == 0: return

    arr = [0] * (n + 1)
    
    def gen(t, p):
        if t > n:
            if n % p == 0:
                seq = tuple(alphabet[arr[i]] for i in range(1, n + 1))
                if seq[0][0] == 0:
                    yield seq
        else:
            arr[t] = arr[t - p]
            yield from gen(t + 1, p)
            for j in range(arr[t - p] + 1, k):
                arr[t] = j
                yield from gen(t + 1, t)

    yield from gen(1, 1)

# --- Worker Function ---

def worker_task(a: int, shift_pairs: Tuple[Tuple[int, int], ...], outdir: str) -> bool:
    adj = build_adj_list(a, shift_pairs)
    n = 2 * a
    size = mis_z3(adj)
    
    if 0 < size <= (n / 3):
        seq_str = "_".join(f"{s1}-{s2}" for s1, s2 in shift_pairs)
        filename = f"a{a}_seq_{seq_str}.adjlist"
        filepath = Path(outdir) / filename
        with open(filepath, "w") as f:
            f.write(str({k: sorted(v) for k, v in adj.items()}))
        return True
    return False

# --- Main Runner ---

def run_parallel_generation(a: int, num_workers: int = 30):
    outdir = f"bi_jewelled_output_a{a}"
    os.makedirs(outdir, exist_ok=True)
    
    n = 2 * a
    target_mis = n / 3
    
    # Calculate exact search space size
    alphabet = generate_even_odd_alphabet(a)
    k_total = len(alphabet)
    k_non_zero = sum(1 for s1, s2 in alphabet if s1 != 0)
    
    # Total necklaces minus necklaces that contain NO 0-shifts
    expected_tasks = macmahon_necklaces(a, k_total) - macmahon_necklaces(a, k_non_zero)
    
    print(f"\n{'='*60}")
    print(f"[*] Starting Analysis for a = {a} (n = {n})")
    print(f"[*] Total pairs in alphabet: {k_total}")
    print(f"[*] Expected valid necklaces to process: {expected_tasks:,}")
    print(f"[*] Target Gap >= 1.5 (Target MIS <= {target_mis:.2f})")
    print(f"{'='*60}")

    if expected_tasks == 0:
        print("[!] No tasks to process for this configuration.")
        return

    start_time = time.time()
    results_found = 0
    total_processed = 0
    
    batch_size = num_workers * 100 
    necklace_gen = generate_unique_necklaces(a)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        while True:
            batch = []
            try:
                for _ in range(batch_size):
                    batch.append(next(necklace_gen))
            except StopIteration:
                pass 

            if not batch:
                break
            
            futures = [executor.submit(worker_task, a, seq, outdir) for seq in batch]
            
            for future in futures:
                try:
                    if future.result():
                        results_found += 1
                except Exception as e:
                    pass # Silently handle worker drops to avoid console spam
                total_processed += 1
            
            # --- Progress and ETA Math ---
            try:
                elapsed = time.time() - start_time
                avg_speed = total_processed / elapsed if elapsed > 0 else 0
                percent = (total_processed / expected_tasks) * 100
                
                if avg_speed > 0:
                    remaining_tasks = expected_tasks - total_processed
                    eta_seconds = remaining_tasks / avg_speed
                    eta_str = format_eta(eta_seconds)
                else:
                    eta_str = "??:??"
                    
                print(f"[Progress] a={a:<2} | {total_processed:,}/{expected_tasks:,} ({percent:05.2f}%) "
                      f"| Found: {results_found} | Speed: {avg_speed:5.1f}/s | ETA: {eta_str}")
            except Exception:
                # Catch-all to ensure print formatting errors never crash the run
                print(f"[Progress] a={a} | Processed: {total_processed} | Found: {results_found}")

    print(f"\n[!] Finished a={a}. Total Valid Graphs (Gap >= 1.5): {results_found}")
    print(f"[!] Total Time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    for test_a in range(6, 17):
        run_parallel_generation(a=test_a, num_workers=30)
