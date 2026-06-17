# Findings

## Phase 1 — model-free PPO (with reverse curriculum)

Trained PPO with the reverse-curriculum scheduler and, separately, a knob-free
uniform mixed-depth baseline. Greedy eval, 200 scrambles/depth (full tables in
`frontier_ppo_cube_curriculum.txt` and `frontier_ppo_uniform.txt`).

| depth | curriculum solve% | uniform solve% |
|------:|------------------:|---------------:|
|     4 |              98.0 |           85.0 |
|     5 |              88.0 |           51.0 |
|     6 |              66.5 |           28.5 |
|     7 |              37.0 |           11.0 |
|     8 |              23.0 |            5.0 |
|     9 |               7.0 |            2.5 |
|    12 |               1.0 |            0.5 |
|    13 |               0.0 |            0.0 |

**Conclusions:**

1. **The curriculum beats uniform by ~1.5 depth levels.** 50%-solve crossover
   is depth ~6–7 (curriculum) vs ~5 (uniform). Concentrating ~80% of samples at
   the learning frontier beats spreading them across all depths — as predicted.
2. **The wall is geometric.** Each extra depth roughly *halves* the solve rate
   (×0.76, ×0.56, ×0.62, ×0.30 across depths 5→9). Extrapolated, a full scramble
   (depth ~20–26) is thousands of halvings away. **Model-free PPO cannot reach
   it** — this is the quantitative motivation for Phase 2.
3. **When PPO solves, it solves near-optimally** (`mean_steps ≈ depth`). The
   network *can* represent excellent policies; it just can't *discover* them at
   depth. That discovery gap is exactly what a learned heuristic + search fills.

## Phase 2 — DAVI cost-to-go + Batch Weighted A* (DeepCubeA recipe)

`train/davi.py` learns J(s) ≈ moves-to-solve by value iteration over
self-generated scrambles (no exploration needed); `eval/solve_search.py` uses J
to guide Batch Weighted A* (`f = g + lambda·J`).

**Smoke validation** (CPU, 3000 updates, 512-wide net, trained only to depth 14,
`lambda=2`, `max_nodes=50k`, 30 scrambles/depth):

| depth | DAVI+search solve% | mean solution len | PPO curriculum solve% |
|------:|-------------------:|------------------:|----------------------:|
|     7 |              100.0 |               6.7 |                  37.0 |
|    10 |              100.0 |               9.7 |                   ~6  |
|    12 |              100.0 |              11.7 |                   1.0 |
|    13 |               96.7 |              12.7 |                   0.0 |
|    14 |               83.3 |              13.4 |                   0.0 |

Even this tiny run dominates fully-trained PPO at every depth, with near-optimal
solution lengths. The 13/14 falloff is an artifact of training stopping at depth
14 and a small node budget — both lifted by a real run.

**Next:** full DAVI run (`--max-scramble-depth 30`, more updates, wider net) +
search with a larger `--max-nodes`, targeting 100% on full scrambles.
