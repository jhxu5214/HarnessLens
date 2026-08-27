# Environment Setup

Benchmark data and agent runtimes are external dependencies and are not
distributed with HarnessLens. Place checkouts under `third_party/`, or symlink
existing local checkouts there. Run `scripts/check_env.py` after setup; it
validates the exact files and executables required by the selected environment.

## Pinned benchmark revisions

| Local path | Upstream repository | Revision |
| --- | --- | --- |
| `third_party/tau3-bench` | <https://github.com/sierra-research/tau2-bench> | `d8e915f7f46b56af9b14d5d0544ccc9fd5d71009` |
| `third_party/bird-mini-dev` | <https://github.com/bird-bench/mini_dev> | `b3d4bcbbae9a96934ad812551eb400c7a3b23c12` |
| `third_party/terminal-bench` | <https://github.com/harbor-framework/terminal-bench> | `1a6ffa9674b571da0ed040c470cb40c4d85f9b9b` |

The `tau3-bench` directory name is retained for path compatibility; the checkout
contains tau2-bench.

Clone and pin all three benchmark repositories from the HarnessLens root:

```bash
mkdir -p third_party
git clone https://github.com/sierra-research/tau2-bench.git third_party/tau3-bench
git -C third_party/tau3-bench checkout d8e915f7f46b56af9b14d5d0544ccc9fd5d71009
git clone https://github.com/bird-bench/mini_dev.git third_party/bird-mini-dev
git -C third_party/bird-mini-dev checkout b3d4bcbbae9a96934ad812551eb400c7a3b23c12
git clone https://github.com/harbor-framework/terminal-bench.git third_party/terminal-bench
git -C third_party/terminal-bench checkout 1a6ffa9674b571da0ed040c470cb40c4d85f9b9b
```

## Retail and Banking Knowledge

Create the tau2 virtual environment inside its checkout:

```bash
cd third_party/tau3-bench
python3 -m venv .venv
.venv/bin/pip install -e .
```

Retail requires its task and policy files. Banking Knowledge additionally
requires its database, document corpus, and retrieval configuration. The
rollout worker and MCP bridge use the checkout's virtualenv, not the HarnessLens
virtualenv.

## Terminal-Bench 2.0

Terminal-Bench requires its task checkout and a reachable Docker daemon. Each
task executes in an isolated container. Set `DOCKER_HOST` for a standard daemon,
or configure the rootless daemon through `HAI_DOCKER_ROOT` / `HAI_DOCKER_HOST`.

Containers must reach the model provider. Environments that require a proxy can
configure the `TB_*` proxy variables documented in
[configuration.md](configuration.md). A small number of verifier files missing
from sparse upstream checkouts are retained under `assets/terminal_task_assets/`
with source records.

## BIRD Mini-Dev

BIRD requires the prompt JSONL plus the downloaded SQLite databases:

```text
third_party/bird-mini-dev/finetuning/inference/mini_dev_prompt.jsonl
third_party/bird-mini-dev/data/dev_databases/<db_id>/<db_id>.sqlite
```

The database files are distributed separately by the benchmark project.

## Agent runtimes

Install at least one target harness:

| Harness | Resolution |
| --- | --- |
| OpenCode | `PATH`, `OPENCODE_PREFIX`, or `HAI_OPENCODE_BIN` |
| Codex CLI | `PATH`; optional model cache via `HAI_CODEX_MODELS_CACHE` |
| Pi | `PI_AGENT_BIN`, or `.pi-agent/node_modules/.bin/pi` |

OpenCode and Codex documentation snapshots used during harness exploration are
vendored under `assets/docs_cache/`. Pi documentation is read from its installed
npm package.

## Isolation

tau2 and BIRD use per-task local workspaces with ground-truth data and grading
outside the agent sandbox. Terminal-Bench uses its task container as the
sandbox. Use the preflight checker before every new environment/harness pair:

```bash
.venv/bin/python scripts/check_env.py --cell retail --harness opencode
```

[中文版本](benchmarks_zh.md)
