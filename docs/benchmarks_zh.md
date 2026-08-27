# 基准数据准备

基准数据与 agent 运行时**不随本仓库分发**：体积大（BIRD 的 sqlite 库就有数 GB）、
许可独立、且各自有自己的虚拟环境。本仓库只依赖它们出现在 `third_party/` 下的
约定位置。

`third_party/` 已被 gitignore。你可以直接 clone 进去，也可以软链到机器上已有的
checkout：

```bash
mkdir -p third_party
ln -s /data/benchmarks/tau2-bench third_party/tau3-bench
```

配好之后先体检，再跑实验：

```bash
.venv/bin/python scripts/check_env.py --cell <cell>
```

---

## 版本固定

下面是产出已报告结果时使用的确切 commit。换 commit 会改变任务定义，也就会改变
split 的含义与分数，务必先确认。

| 目录 | 仓库 | Commit |
| --- | --- | --- |
| `third_party/tau3-bench` | <https://github.com/sierra-research/tau2-bench> | `d8e915f7f46b56af9b14d5d0544ccc9fd5d71009` |
| `third_party/bird-mini-dev` | <https://github.com/bird-bench/mini_dev> | `b3d4bcbbae9a96934ad812551eb400c7a3b23c12` |
| `third_party/terminal-bench` | <https://github.com/harbor-framework/terminal-bench> | `1a6ffa9674b571da0ed040c470cb40c4d85f9b9b` |

> `tau3-bench` 这个目录名是历史遗留：里面装的是 **tau2-bench**。代码里的路径常量
> 用的就是这个名字，改名会破坏 baseline 指纹，所以保持不变。

```bash
mkdir -p third_party
git clone https://github.com/sierra-research/tau2-bench third_party/tau3-bench
git -C third_party/tau3-bench checkout d8e915f7f46b56af9b14d5d0544ccc9fd5d71009
git clone https://github.com/bird-bench/mini_dev.git third_party/bird-mini-dev
git -C third_party/bird-mini-dev checkout b3d4bcbbae9a96934ad812551eb400c7a3b23c12
git clone https://github.com/harbor-framework/terminal-bench.git third_party/terminal-bench
git -C third_party/terminal-bench checkout 1a6ffa9674b571da0ed040c470cb40c4d85f9b9b
```

---

## 分 cell 配置

### `retail` / `banking`（τ²-bench）

需要 tau2 的 checkout 加上它自己的虚拟环境——rollout worker 和 MCP bridge 都是
用**这个 venv 的解释器**起的子进程，而不是编排器的 venv。

```bash
cd third_party/tau3-bench
python3 -m venv .venv          # 或者按上游 README 用 uv
.venv/bin/pip install -e .
```

需要的文件：

```
third_party/tau3-bench/.venv/bin/python3
third_party/tau3-bench/src/                                     # PYTHONPATH
third_party/tau3-bench/data/tau2/domains/retail/tasks.json
third_party/tau3-bench/data/tau2/domains/retail/policy.md
third_party/tau3-bench/data/tau2/domains/retail/split_tasks.json
third_party/tau3-bench/data/tau2/domains/banking_knowledge/tasks.json
third_party/tau3-bench/data/tau2/domains/banking_knowledge/db.json
third_party/tau3-bench/data/tau2/domains/banking_knowledge/documents/
```

banking 还需要 `HAI_TAU2_RETRIEVAL_CONFIG`（默认 `bm25`，不能为空）。

### `terminal-bench`

需要任务定义加一台**能用的 Docker 主机**——每个任务一个容器，容器就是沙箱。

```
third_party/terminal-bench/original-tasks/<task-id>/task.yaml     # 或 task.toml + instruction.md
third_party/terminal-bench/original-tasks/<task-id>/run-tests.sh  # 或 tests/test.sh
```

个别任务的 sparse checkout 里缺 verifier 文件。这类文件已经 vendored 在
`assets/terminal_task_assets/<task-id>/`，运行时会在 agent 跑完之后拷进容器；
出处记在各自的 `SOURCE.md` 里。

容器里的 agent 需要连 provider。如果这条链路要走代理，配好
`TB_CLASHCTL_HOME` 与 `TB_CONTAINER_PROXY_URL`（见
[configuration_zh.md](configuration_zh.md)），blind TEST 基线走
`scripts/run_test_baseline_clash.sh`。

rootless docker 用 `HAI_DOCKER_ROOT` / `HAI_DOCKER_HOST` 指定；标准 docker 直接
设 `DOCKER_HOST` 即可。

### `bird`（BIRD Mini-Dev，challenging 子集）

```
third_party/bird-mini-dev/finetuning/inference/mini_dev_prompt.jsonl
third_party/bird-mini-dev/data/dev_databases/<db_id>/<db_id>.sqlite
```

sqlite 库要按上游 README 单独下载，不在 git 仓库里。TRAIN/TEST split 冻结在
`configs/bird_mini_dev_challenging_split.json`（seed 42，102 题里抽 30 TRAIN /
72 TEST）。

---

## Agent 运行时

被迭代的 harness 本身也是外部依赖。至少装一个：

| harness | 安装方式 | 定位方式 |
| --- | --- | --- |
| **opencode** | 官方安装脚本 | `PATH`，否则 `OPENCODE_PREFIX`（默认 `~/.opencode`），否则 `HAI_OPENCODE_BIN` |
| **codex** | 官方 CLI | `PATH`；模型目录缓存来自 `HAI_CODEX_MODELS_CACHE`（默认 `~/.codex/models_cache.json`） |
| **pi** | `npm install @earendil-works/pi-coding-agent` 到 `<repo>/.pi-agent` | `PI_AGENT_BIN`，否则 `<repo>/.pi-agent/node_modules/.bin/pi` |

pi 还有一处额外作用：Harness Query 会读 npm 包自带的文档
（`.pi-agent/node_modules/@earendil-works/pi-coding-agent/docs/`）作为
`local_documentation` 证据。没装 pi 时该证据缺失，对应的测试会 skip。

opencode 与 codex 的官方文档快照已经 vendored 在 `assets/docs_cache/`，无需联网。

---

## 隔离说明

tau2 的 rollout 采用本地无根隔离（`local_rootless_rollout=True`）：
每个任务独立工作目录，GT 文件锁在沙箱外，verifier 在沙箱外校验。
Terminal-Bench 不走这条路径——容器本身就是沙箱。
