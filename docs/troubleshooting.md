# Troubleshooting

Start with the environment checker:

```bash
.venv/bin/python scripts/check_env.py --cell <cell> --harness <harness>
```

## Run cannot resume

`workflow source changed during the run` means that source files participating
in the workflow fingerprint changed after the run started. Restore the matching
source or start a new run ID.

`baseline event runtime fingerprint differs from this run` means that the saved
baseline was produced by different runtime code, tasks, or harness settings.
Generate a fresh baseline instead of forcing reuse.

## Runtime is unavailable

If OpenCode, Codex, or Pi cannot be found, install the selected runtime and
verify the path variables in [configuration.md](configuration.md). For tau2,
`ModuleNotFoundError: tau2` usually means the checkout-local virtualenv was not
created or the checkout is at the wrong path.

## Terminal-Bench and Docker

If the Docker socket is missing, run the environment checker and verify
`DOCKER_HOST` or `HAI_DOCKER_ROOT`. A mismatch between the rootless state
directory and socket is rejected deliberately to prevent the run from using an
empty daemon state.

Interrupted Terminal-Bench runs may leave containers behind. Inspect first,
then use `scripts/reap_containers.py` to remove HarnessLens-owned orphans.

## Candidate is not promoted

Messages about missing attributable evidence, residual probes, or failed
channel preflight are decision outcomes rather than infrastructure errors. The
candidate either lacks trajectory-supported improvement, exposes a regression,
or did not load the edited channel in the effective runtime. Review artifacts
under `runs/train/<run-id>/main_agent/` and `rollout_evidence/`.

## Provider throttling

Reduce rollout and analysis concurrency through the documented `HAI_*`
concurrency variables. Provider failures are recorded in API traces; do not
reinterpret infrastructure failures as task failures.

## Disk usage

Trajectory workspaces and container caches can be large. Keep
`HAI_KEEP_TRAJECTORY_WORKSPACE=0` for normal runs and configure
`HAI_MIN_FREE_GB` as a pre-launch guard when storage is constrained.

## Tests

Live tests require installed runtimes, provider credentials, and in some cases
Docker. The default `scripts/run_tests.sh` suite excludes them; pass `--live`
only on a fully configured machine.

[中文完整参考](troubleshooting_zh.md)
