#!/usr/bin/env python3
"""Exhaustive small-schema verification of the residual-failure-path
lemma (main-text Lemma on the failure path):

    q not in Cn(B~ u S)   <=>   Dexp(q, S) not subseteq B~

Semantics implemented verbatim from the paper (Section "Availability"):
  - schema G_q is a DAG; the first L vertices are premise leaves
    (no parents); every later vertex is internal with >=1 parents
    among earlier vertices; the last vertex is the service sink q;
  - Cn_q(X) is the least set containing X n V(G_q) and closed under
    the rule  pa(v) subseteq C => v in C  for internal v only;
  - U(q,S) = S n V(G_q); Dexp(q,S) is the set of premise leaves with
    a cache-free path (no vertex of U(q,S)) to the sink;
  - B~ ranges over ALL subsets of the premise leaves; S ranges over
    ALL subsets of V(G_q) \ {sink} (cached support vertices, incl.
    possibly leaves; service sinks excluded by definition).

For every generated schema the biconditional is checked on the full
state space 2^leaves x 2^(|V|-1).  Any mismatch is reported with a
witness.  Output: dag_correctness_results.json.
"""
import json
import os
import random
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261022


def gen_schema(rng, n_leaves, n_internal):
    """Vertices 0..L-1 leaves; L..L+I internal; sink = L+I-1 (last).
    Every internal vertex picks 1..3 parents among earlier vertices;
    internal vertex L+I-2..: ensure the sink is reachable by giving
    the last few vertices parents near the end.  Returns parents list."""
    L, I = n_leaves, n_internal
    parents = [[] for _ in range(L + I)]
    for v in range(L, L + I):
        k = rng.randint(1, min(3, v))
        parents[v] = rng.sample(range(v), k)
    # force the sink to depend on the final internal layers so the
    # schema is not degenerate
    sink = L + I - 1
    if I >= 2:
        extra = [sink - 1]
        for p in extra:
            if p not in parents[sink]:
                parents[sink].append(p)
    return parents


def closure(parents, X):
    """Least closure: X plus internal vertices whose parents are all in."""
    n = len(parents)
    C = set(X)
    changed = True
    while changed:
        changed = False
        for v in range(n):
            if v in C or not parents[v]:
                continue  # leaves enter only via X
            if all(p in C for p in parents[v]):
                C.add(v)
                changed = True
    return C


def exposed_leaves(parents, n_leaves, S):
    """Leaves with a path to the sink avoiding every vertex of S."""
    n = len(parents)
    sink = n - 1
    if sink in S:
        raise ValueError("sink must not be cached")
    # reverse reachability from the sink over vertices NOT in S:
    # walk PARENT edges backward (which vertices can reach the sink)
    reachable = set()
    stack = [sink]
    while stack:
        v = stack.pop()
        if v in reachable or v in S:
            continue
        reachable.add(v)
        stack.extend(parents[v])
    return {b for b in range(n_leaves) if b in reachable}


def subsets(iterable):
    items = sorted(iterable)
    for r in range(len(items) + 1):
        for c in combinations(items, r):
            yield set(c)


def check_schema(parents, n_leaves):
    n = len(parents)
    sink = n - 1
    cacheable = [v for v in range(n) if v != sink]
    states = 0
    for S in subsets(cacheable):
        Dexp = exposed_leaves(parents, n_leaves, S)
        for Bt in subsets(range(n_leaves)):
            states += 1
            X = Bt | S
            certified = sink in closure(parents, X)
            rhs = Dexp <= Bt  # Dexp subseteq B~
            # lemma: NOT certified  <=>  NOT rhs
            if certified != rhs:
                return states, (S, Bt, certified, rhs)
    return states, None


def main():
    rng = random.Random(SEED)
    total_states = 0
    total_schemas = 0
    configs = []
    for trial in range(400):
        L = rng.randint(1, 5)
        I = rng.randint(1, 6)
        if 2 ** L * 2 ** (L + I - 1) > 200000:
            continue
        parents = gen_schema(rng, L, I)
        states, witness = check_schema(parents, n_leaves=L)
        total_states += states
        total_schemas += 1
        configs.append({"leaves": L, "internal": I, "states": states})
        if witness is not None:
            out = {"result": "MISMATCH", "trial": trial,
                   "leaves": L, "internal": I,
                   "parents": parents,
                   "witness_S": sorted(witness[0]),
                   "witness_Bt": sorted(witness[1]),
                   "certified": witness[2], "rhs": witness[3]}
            json.dump(out, open(os.path.join(
                HERE, "dag_correctness_results.json"), "w"), indent=1)
            print("MISMATCH:", json.dumps(out))
            return
    out = {
        "result": "all states agree",
        "seed": SEED,
        "schemas_tested": total_schemas,
        "states_checked": total_states,
        "max_leaves": max(c["leaves"] for c in configs),
        "max_vertices": max(c["leaves"] + c["internal"] for c in configs),
        "statement": ("q not in Cn(B~ u S) <=> Dexp(q,S) not subseteq B~ "
                      "verified exhaustively over all erasure/cache states "
                      "of each randomly generated conjunctive schema"),
    }
    json.dump(out, open(os.path.join(HERE, "dag_correctness_results.json"),
                        "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
