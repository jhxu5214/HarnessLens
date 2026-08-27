# Canonical split references

Frozen copies of the split definitions that HarnessLens' own
`configs/*_split.json` files were derived from. They exist so
`tests/test_blind_test_eval.py::test_owned_splits_match_the_canonical_inputs`
can keep proving that the shipped splits never drifted from the seed-42
draw that produced the reported numbers.

| File | Origin |
| --- | --- |
| `banking_knowledge_split.json` | tau2 `banking_knowledge` TRAIN/TEST draw, seed 42, 30/67 |
| `terminal_bench_split_seed42_train30_test59.yaml` | Terminal-Bench TRAIN/TEST draw, seed 42, 30/59 |

The retail reference is not mirrored here: it ships with the benchmark itself at
`third_party/tau3-bench/data/tau2/domains/retail/split_tasks.json`.

These files are reference data only. Nothing in the runtime reads them.
