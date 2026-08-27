# Architecture

HarnessLens evolves user-configurable harness state while keeping the base
model, harness framework, task tools, permissions, and evaluation protocol
fixed. A deterministic controller coordinates model-based roles and task
rollouts, accounts for every interaction, and isolates TRAIN from blind TEST.

## Workflow

```text
baseline rollout
      |
      v
Context Exploration
  - Task-Space Exploration
  - Harness-Space Exploration
      |
      v
Trajectory Diagnosis
  - Experience Extraction
  - Experience Analysis
      |
      v
Harness Evolution
  - Candidate Proposal
  - Behavior-Aware Verification
  - Harness Review and Update
      |
      v
submission/final.json
```

The controller in `harnesslens/evolution/controller.py` checkpoints every stage.
A run with the same `--run-id` resumes only when its workflow and baseline
runtime fingerprints still match the recorded source.

## Core modules

| Module | Responsibility |
| --- | --- |
| `evolution/discovery.py` | Organize TRAIN tasks and discover editable harness surfaces from documentation and runtime probes |
| `evolution/experience.py` | Extract reusable behavior and recurring deficiencies from retained trajectories |
| `evolution/analyzer.py` | Turn trajectory evidence into attributable modification proposals |
| `evolution/main_agent.py` | Select proposals, choose verification tasks, review evidence, and decide whether to update |
| `evolution/harness_editor.py` | Materialize a proposed change in an isolated harness workspace |
| `harnesses/channel_preflight.py` | Verify that each edited channel is visible in the effective runtime |
| `evolution/rollout.py` / `evaluation/rollout_bridge.py` | Run matched incumbent/candidate trials and retain evidence |

Harness adapters map the common workflow onto OpenCode, Codex CLI, and Pi.
Benchmark adapters provide tau2, Terminal-Bench, and BIRD execution and grading.

## Verification and promotion

Each candidate starts from the current confirmed harness. Verification tasks
are selected for the targeted behavior, related task patterns, and regression
risk. Candidate and parent run under matched conditions. A candidate can advance
only when a strict paired metric gain is supported by attributable trajectory
evidence and no attributable regression is observed. A separate confirmation
batch is required before promotion.

If these conditions are not met, the parent remains current. Accepted edits can
accumulate because the next candidate is derived from the latest confirmed
harness.

## Budget and artifacts

`harnesslens/core/budget.py` maintains an auditable ledger of task trials and
complete model-based sessions. The controller reserves enough budget for a
complete verification cycle before launching it.

Run artifacts live under `runs/train/<run-id>/`:

| Path | Contents |
| --- | --- |
| `controller_state.json` | resumable stage checkpoint |
| `creation_budget.json` | interaction ledger |
| `discovery/`, `experience/`, `analyzer/` | analysis inputs and outputs |
| `main_agent/` | candidate and review decisions |
| `rollout_evidence/` | retained trajectories and API traces |
| `submission/final.json` | selected harness and effective manifest |

Blind TEST uses separate entrypoints and artifact roots. TEST tasks, services,
trajectories, and feedback are unavailable to every TRAIN-stage role.

[中文版本](architecture_zh.md)
