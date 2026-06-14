# Results

Plain-text experiment outputs committed here so they can be read/reviewed in
the repo (handy when copy-paste out of the training environment is blocked).

Checkpoints themselves stay out of git (`runs/` is git-ignored) — only the
text tables/logs live here.

## How to add a result

Pipe any eval or training output straight to a file with `tee`, then commit:

```bash
# Frontier curve for a trained checkpoint
python -m eval.policy_eval --checkpoint runs/<run_name>/agent.pt --max-depth 14 \
    | tee results/frontier_<run_name>.txt

# (optional) also capture the training tail / curriculum promotion log
python -m train.ppo_cube ... | tee results/train_<run_name>.log

git add results/
git commit -m "Add eval results for <run_name>"
git push
```

## Naming

`frontier_<run_name>.txt` for eval tables, `train_<run_name>.log` for training
logs. Include the command you ran at the top of the file if it isn't obvious
from the name, so the result is reproducible.
