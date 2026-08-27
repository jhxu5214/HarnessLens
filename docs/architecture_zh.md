# 架构

## 0. 实验约束

整套系统建立在一个前提上：**除了 harness，什么都不许变。**

固定不变的有：模型、provider、reasoning level、任务侧工具、权限、预算、评测方式、
TRAIN/TEST split。可变的只有 harness 自身的可编辑通道——system prompt、
project instructions、工具描述、MCP 工具 schema、skills、agent 定义、hooks、
运行时 config。

因此任何分数变化都只能归因到 harness。这也是所有模块 prompt 里反复强调"不要
推断 grader 逻辑 / 隐藏答案 / reward 成因"的原因：一旦候选 harness 里出现了对具体
任务的记忆，改进就不再是 harness 层面的改进，而是泄题。

---

## 1. 控制器

`harnesslens/evolution/controller.py` 的 `IterationController` 按固定顺序驱动五个模块，
每步写 checkpoint 到 `runs/train/<run-id>/controller_state.json`。

```
ensure_baseline_event      基线 rollout（或复用已有 baseline event）
  └ checkpoint: baseline_ready
DiscoveryModule
  └ checkpoint: discovery_complete
ExperienceModule.run_baseline
  └ checkpoint: experience_complete
AnalyzerModule
  └ checkpoint: analyzer_complete
MainAgentModule  ──┐
  ├ channel preflight
  ├ 配对 screen rollout
  ├ 配对 confirm rollout（视 evaluation_mode 而定）
  └ 晋升判定 → 循环，直到预算耗尽或无可迭代候选
submission/final.json + CURRENT_HEAD.txt
```

### workflow fingerprint

run 开始时，`core/workflow_fingerprint.py` 会对 `run_e2e.py` 加 `harnesslens/**/*.py`
做整体 SHA256，写入 `runs/train/<run-id>/workflow_fingerprint.json`。此后每个
阶段边界都会重新校验。**run 进行中改动源码会直接让这次 run 失败**，必须换新
run-id——这是为了保证一次 run 的产物出自同一份代码。

同理，`evolution/baseline.py` 里的 `BASELINE_RUNTIME_FILES` / `BenchmarkConfig.runtime_files()`
按 cell 列出参与 rollout 的运行时文件，baseline event 记录它们的指纹；指纹不匹配
的 baseline 不能复用。

---

## 2. 五个模块

### Discovery（`evolution/discovery.py`）

两个探索角色，各自独立：

- **Task Explorer**：只看 TRAIN 任务定义，按用户目标分类，最多十个主类。不推断
  期望动作、grader 逻辑、隐藏答案。产出是后续归因的组织框架，不是结论。
- **Harness Query**：拿 architecture probe（真实启动一次目标运行时抓到的行为）、
  官方文档快照（`assets/docs_cache/`）、运行时观测、候选工作区契约，判定
  **这个 harness 在这个实验里究竟能改什么**。

Harness Query 的判定标准很严：一个通道算"当前可改"，必须同时满足——候选能用
`home/` 或 `project/` 下的文件实现这个机制、固定的运行时确实会加载它、有证据支持。
文档里写了但被固定配置覆盖、只能靠 CLI 开关、或者功能本身被禁用的，都算
conditional 或 unavailable，不算可改。

各 harness 的适配器在 `harnesses/harness_query_adapters.py`（codex / pi 的静态
native 适配器）与 `harnesses/opencode_harness.py`（opencode）。

### Experience（`evolution/experience.py`）

读全部 TRAIN 轨迹，产出两类自然语言段落：

- `reusable`：以成功为主的行为。保留具体上下文、动作顺序、工具与参数细节、
  观察到的结果、确认与分支逻辑。只有明确等价的路径才合并；成功分支有实质差异
  时保留为不同段落。**不允许把具体流程抽象成空泛原则。**
- `needs_adjustment`：反复出现的具体失败。只有当"可观察触发条件"与"问题行为"
  都匹配时才合并失败。单条失败轨迹只是一个有界观察，不构成 reward 成因；当同一
  任务的所有对比 trial 都失败时，该模式标为 unresolved，只陈述现象不下结论。

`scripts/stages/run_experience_comparison.py` 对应的比较模式则用于候选 vs 现任的差异分析。

### Analyzer（`evolution/analyzer.py`）

把 Experience 的段落归因到**能承载它的最窄通道**，产出带排序的候选组合。核心
判据是通道语义：可见时机、作用范围、约束强度、触发机制、回归风险。

- 选工具的提示 → 该工具的描述
- 选参数的约束 → 该参数的描述
- 有分支、参数、检查、恢复逻辑的有序流程 → on-demand skill
- 简洁的普适策略 → instructions
- 功能启用 → config

每个候选只承载**一个原子行为假设**，一个可观察触发条件、一个可观察结果。它可以
同时动多个通道，但前提是这些改动共同实现同一个假设；相互独立的流程必须拆成不同
候选。

`REUSABLE_*` 与 `ADJUSTMENT_*` 两套 prompt 分别处理两类经验，`POST_*` 版本用于
一次 rollout 之后的复盘归因。

### Main Agent（`evolution/main_agent.py`）+ Harness Editor（`evolution/harness_editor.py`）

Main Agent 负责选候选、排预算、决定评测模式、读 review 结果决定下一步
（accept / revise / replan / defer）。

Harness Editor 是真正动手改的角色：拿 Harness Query 报告，直接编辑隔离的
`home/` 与 `project/` 树。约束写在 `EDITOR_SYSTEM_PROMPT` 里，最关键的一条是
**候选产物里不得出现经验 ID、任务 ID、基准实体、字面答案或逐例映射**——学到的
东西必须表达成可复用的原则。

各 harness 的 `home/` 作用域：

| harness | home 作用域 |
| --- | --- |
| opencode | `XDG_CONFIG_HOME/opencode` |
| pi | `PI_CODING_AGENT_DIR` |
| codex | `CODEX_HOME` |

### Channel Preflight（`harnesses/channel_preflight.py`）

落地之后、花钱跑 rollout 之前，先花 1 个 creation 跑一次 preflight：验证被改的
每个通道**确实被运行时加载了**。改了 skill 但运行时没启用 skill 工具、改了
config 但被固定覆盖顶掉——这些在 preflight 就会被拦下，而不是浪费一整轮
rollout 才发现。失败会以 `ChannelPreflightError` 终止本候选并记录报告。

### Rollout（`evolution/rollout.py` / `evaluation/rollout_bridge.py`）

配对 rollout：候选与现任冠军在**同一批任务、同一 seed、同一 trial 数**下跑，
逐任务配对比较。`rollout_bridge.py` 负责与各基准的实际执行器对接，并保留每个
trial 的完整轨迹与 API trace（`validate_rollout_interactions` 会拒收缺少完整
trace 的记录）。

---

## 3. 晋升协议

```
screen  → 低成本初筛，至少 MIN_STANDARD_ROLLOUT_TASKS = 5 个任务
   │
   ├─ decision = accept_delta 且 evaluation_mode 需要复验
   │     └→ confirm：另一批任务上的独立第二次配对
   │           └→ 两次都赢 → 晋升，写入 CURRENT_HEAD.txt
   │           └→ 否则     → reject_unconfirmed_candidate
   │
   ├─ evaluation_mode = residual_probe 且 accept_delta
   │     └→ defer_residual_candidate（证据不足以支撑完整复验，挂起）
   │
   └─ 其它 → revise / replan，回到 Main Agent
```

三种 `evaluation_mode`：

| 模式 | 含义 |
| --- | --- |
| `standard` | 常规：screen + confirm |
| `terminal_screen` | 预算只够 screen，不做 confirm（此时不允许晋升） |
| `residual_probe` | 任务数低于常规下限（≥ `MIN_RESIDUAL_ROLLOUT_TASKS = 2`）的探针 |

关键不变量，都由 `controller.py` 里的断言强制：

- `assert_candidate_extends_champion`：候选必须以当前冠军为父版本，不能从旧分支冒出来。
- `assert_confirmed_promotion`：没有 confirm 通过就不能晋升。
- `advance_cumulative_version`：版本号只能沿冠军链累进。

`HAI_CONFIRMATION_MODE=off` 可以关掉 confirm，但那样跑出来的结果只能当探索，
不能当结论。

---

## 4. Creation budget（`budget.py`）

一次 run 能开启的"隔离智能会话"总数是硬上限，默认 200
（`HAI_TOTAL_CREATION_BUDGET`）。

- 基线的 `30 × TRAIN_ROLLOUT_REPEATS` 次会话在 run 开始时就预扣为 `baseline_used`。
- 之后每一次智能会话单独计费，带 reason 落盘到 `creation_budget.json`。
- 预算文件用 `fcntl` 加锁，进程被杀掉后重启会调用
  `recover_interrupted_jobs()` 回收未结算的会话。
- Main Agent 在开新一轮之前会先算这一轮的完整成本
  （preflight + screen + confirm + `ITERATION_RETRY_BUFFER_CREATIONS = 3` 的重试缓冲）；
  预算不够就降级评测模式或直接收尾，不会开一轮做到一半没钱。

---

## 5. 产物布局

```
runs/
├── train/<run-id>/                     TRAIN 自迭代
│   ├── workflow_fingerprint.json       本次 run 的源码指纹
│   ├── controller_state.json           阶段 checkpoint（恢复用）
│   ├── creation_budget.json            会话计量账本
│   ├── baseline/                       baseline event 及复用记录
│   ├── discovery/                      task 分类 + harness query 报告
│   ├── experience/                     reusable / needs_adjustment 段落
│   ├── analyzer/                       归因后的候选组合
│   ├── main_agent/                     每轮决策、preflight 报告、review
│   ├── harness_editor/                 每个候选的工作区快照与 diff
│   ├── rollout_evidence/               逐 trial 轨迹 + API trace
│   ├── submission/final.json           最终选定版本
│   └── CURRENT_HEAD.txt                冠军指针
├── test_baselines/<run-id>/<harness>/  blind TEST 基线
└── test_candidates/<run-id>/<harness>/ blind TEST 候选
```

TRAIN 与 TEST 是两棵完全独立的树。控制器只写 `train/`，只读 `train/`。

---

## 6. 支持的 harness

支持的 harness 恰好是 `--harness` 开放的三个，没有别的：

| harness | 适配器 | tau2 驱动 | BIRD 驱动 |
| --- | --- | --- | --- |
| opencode | `opencode_harness.py` / `opencode_runtime.py` | `opencode_tau2.py` | `bird_eval.py` |
| codex | `harness_query_adapters.py` | `codex_tau2.py` + `codex_responses_proxy.py` | `bird_eval.py` |
| pi | `harness_query_adapters.py` | `pi_tau2.py` + `pi_compact_runner.mjs` | `bird_eval.py` |

### 两个共享驱动层

跨 harness 的 benchmark 执行由两个共享驱动层承载：

| 模块 | 谁依赖它 | 它做什么 |
| --- | --- | --- |
| `tau2_driver.py` | retail / banking × 三个 harness | tau2 环境装配、工具定义、user simulator socket、trial 行读写 |
| `native_harness_driver.py` | BIRD | 拉起 harness 二进制、运行 provider 代理，并把各家 session 格式解析成 `HarnessTurn` |

两个模块都参与对应 benchmark 的 runtime fingerprint。
