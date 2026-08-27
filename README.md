# HarnessLens

**Verify Smarter, Evolve Further: Efficient Harness Evolution through
Behavior-Aware Verification**

This repository provides the paper implementation for OpenCode, Codex CLI, and
Pi Coding Agent, including the full TRAIN evolution loop and separate blind-TEST
evaluation entrypoints.

[中文文档](README_zh.md)

## Repository layout

```text
harnesslens/
  core/               configuration, budgets, persistence, and shared protocols
  evolution/          controller and the diagnosis/evolution stages
  harnesses/          editable surfaces, manifests, and agent-runtime adapters
  benchmarks/         benchmark task drivers, MCP servers, and grading logic
  evaluation/         rollout services and blind-TEST evaluation
  infrastructure/     process, proxy, provider, and container utilities
run_*.py              public full-loop and blind-TEST entrypoints
configs/              runtime benchmark configurations
assets/               vendored harness documentation and canonical split metadata
scripts/              setup, preflight, advanced stage tools, and maintenance
tests/                offline tests plus three live-runtime suites
third_party/          external benchmark checkouts (not distributed here)
runs/                 generated run artifacts (gitignored)
```

## Requirements

| | |
| --- | --- |
| Python | 3.11+ |
| Python packages | `pyyaml`, `httpx`, `loguru`, and `pytest` |
| Provider | An OpenAI-compatible chat-completions endpoint |
| Agent runtime | At least one of **opencode**, **codex**, or **pi** |
| Benchmarks | Checkouts listed in [docs/benchmarks.md](docs/benchmarks.md), placed or symlinked under `third_party/` |
| Terminal-Bench only | A working Docker host, with one container per task |

The paper uses `deepseek-v4-flash-preview` for all target-agent and evolution
roles, OpenCode v1.17.13, Codex CLI v0.144.4, and Pi Coding Agent v0.80.10.
Benchmark data and agent runtimes are not vendored because they are large,
separately licensed dependencies; the reproducibility document pins their
versions and setup.

## Install

```bash
git clone https://github.com/jhxu5214/HarnessLens.git
cd HarnessLens

# Create .venv, install HarnessLens dependencies, and generate .env.
scripts/setup.sh

# Configure the provider.
$EDITOR .env  # DEEPSEEK_API_KEY; DEEPSEEK_BASE_URL is optional
```

## Environment setup

Benchmark datasets and agent runtimes are external dependencies. Clone them at
the pinned revisions, or symlink existing checkouts into `third_party/`, by
following [docs/benchmarks.md](docs/benchmarks.md). Machine-specific
paths, Docker, and proxy settings are documented in
[docs/configuration.md](docs/configuration.md).

| Environment | Additional local requirement |
| --- | --- |
| Retail / Banking Knowledge | tau2-bench checkout and its virtualenv; Banking also needs its knowledge corpus |
| Terminal-Bench 2.0 | Terminal-Bench checkout and a reachable Docker host |
| BIRD Mini-Dev | BIRD checkout, prompt JSONL, and SQLite databases |

## Quickstart

```bash
.venv/bin/python scripts/check_env.py --cell retail --harness opencode
scripts/run_e2e.sh --run-id retail-001 --cell retail --harness opencode
```

Choose another supported cell or harness with `--cell` and `--harness`. Reusing
the same `--run-id` resumes an interrupted run. Artifacts are written to
`runs/train/<run-id>/`; the selected harness is
`submission/final.json`, and `controller_state.json` records the checkpoint.

### Blind TEST evaluation

Blind TEST is intentionally separate from evolution. Run it only after the
TRAIN controller has produced `submission/final.json`:

```bash
.venv/bin/python run_test_candidate.py --benchmark retail --harness opencode \
  --run-id retail-001-test \
  --patch-json runs/train/retail-001/submission/final.json
```

Use `banking`, `terminal-bench`, or `bird-mini-dev-challenging` as the
`--benchmark` value for the other environments. `run_test_baseline.py` evaluates
the corresponding unmodified harness with the same blind protocol.

## Tests

```bash
scripts/run_tests.sh
```

Pass `--live` to include real runtime and provider checks.

## Documentation

| Document | Contents |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | modules, verification protocol, and artifacts |
| [docs/configuration.md](docs/configuration.md) | environment variables and runtime behavior |
| [docs/benchmarks.md](docs/benchmarks.md) | benchmark checkouts, pinned revisions, and environment setup |
| [docs/troubleshooting.md](docs/troubleshooting.md) | common failures and diagnostics |

## Scope and limitations

The reported evaluation covers one model family, three harnesses, and four
public benchmarks. Interaction units make the budget auditable, but they do not
normalize tokens, latency, or monetary cost across roles and environments.
Behavior-aware evolution is most effective when multiple tasks expose a shared,
controllable behavior; highly diverse task sets may correctly yield no edit
because isolated fixes lack transferable evidence.

## License

HarnessLens is released under the [MIT License](LICENSE). Vendored documentation
under `assets/` retains its upstream terms; benchmark data and agent runtimes are
not distributed here.
