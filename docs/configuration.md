# Configuration

HarnessLens reads configuration in this order, from highest to lowest priority:

```text
exported environment > .env.local > .env > code defaults
```

Existing environment variables are never overwritten by an env file.

## Required provider settings

| Variable | Purpose |
| --- | --- |
| `DEEPSEEK_API_KEY` | Credential used by target-agent rollouts and model-based evolution roles |
| `DEEPSEEK_BASE_URL` | OpenAI-compatible provider endpoint |

Start from `.env.example`; keep credentials in `.env` or `.env.local`, both of
which are ignored by Git.

## Runtime and checkout paths

| Variable | Purpose |
| --- | --- |
| `HARNESSLENS_ROOT` | Override repository root discovery |
| `OPENCODE_PREFIX` / `HAI_OPENCODE_BIN` | Locate OpenCode |
| `PI_AGENT_BIN` / `PI_BIN` | Locate Pi |
| `HAI_CODEX_MODELS_CACHE` | Seed the isolated Codex model cache |
| `TAU2_DATA_DIR` | Override the tau2 data directory |
| `TB_DOCKER` | Override the Docker executable |

Executables already available on `PATH` take precedence where applicable.

## Budget and concurrency

| Variable | Purpose |
| --- | --- |
| `HAI_TOTAL_CREATION_BUDGET` | Total interaction budget for one evolution run |
| `HAI_TRAIN_ROLLOUT_REPEATS` | Repeated TRAIN trials per selected task |
| `HAI_MAX_ROLLOUT_CONCURRENCY` | Concurrent task rollouts |
| `HAI_MAX_ANALYSIS_CONCURRENCY` | Concurrent analysis roles |
| `HAI_PROVIDER_MAX_CONCURRENCY` | Global provider request limit |
| `HAI_PROVIDER_TRIAL_MAX_CONCURRENCY` | Per-trial provider request limit |
| `HAI_PROVIDER_RETRY_ATTEMPTS` | Provider retry limit |
| `HAI_MIN_MEM_GB` / `HAI_MIN_FREE_GB` | Optional resource guards |

Defaults are defined and validated by `harnesslens/core/config.py` and
`harnesslens/core/train_protocol.py`. Experiment-specific settings remain in the
released code and paper rather than being duplicated in the README.

## Evolution behavior

| Variable | Purpose |
| --- | --- |
| `HAI_CONFIRMATION_MODE` | Require or disable independent confirmation before promotion |
| `HAI_BASELINE_REUSE_POLICY` | Control compatibility checks for reused baseline evidence |
| `HAI_KEEP_TRAJECTORY_WORKSPACE` | Retain task workspaces for debugging |
| `HAI_TAU2_TIMEOUT_PER_TURN_S` | Override tau2 per-turn timeout |

Disabling confirmation produces exploratory runs and should not be treated as
the reported protocol.

## Environment-specific settings

Banking Knowledge uses `HAI_TAU2_RETRIEVAL_CONFIG`. Terminal-Bench Docker,
network, image, and proxy settings use `DOCKER_HOST`, `HAI_DOCKER_*`, and
`TB_*`. See [benchmarks.md](benchmarks.md) for the required external files.

## Validate configuration

```bash
.venv/bin/python scripts/check_env.py --cell retail --harness opencode
```

The checker resolves the selected environment rather than testing a generic
dependency list. Any missing required item produces a nonzero exit code.

[中文完整参考](configuration_zh.md)
