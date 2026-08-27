# 配置参考

## 配置来源与优先级

```
已 export 的 shell 环境变量        ← 最高优先级
        ↑
<repo>/.env.local                  （gitignore，放机器特有的覆盖）
        ↑
<repo>/.env                        （gitignore，放主配置）
        ↑
代码内默认值                        ← 最低优先级
```

由 `harnesslens/core/config.py:load_repo_env()` 实现，用的是 `os.environ.setdefault`，
所以**已经存在的环境变量永远不会被 .env 覆盖**。每个入口脚本在解析完参数后立刻
调用它。

从 `.env.example` 复制一份开始：

```bash
cp .env.example .env
```

---

## 必填

| 变量 | 说明 |
| --- | --- |
| `DEEPSEEK_API_KEY` | provider 凭据。所有智能角色 + 所有候选 harness rollout 都用它。 |
| `DEEPSEEK_BASE_URL` | OpenAI 兼容的 chat-completions 端点。解析顺序 `DEEPSEEK_BASE_URL` → `DEEPSEEK_URL`（旧名）→ `https://api.deepseek.com/v1`，统一由 `config.provider_base_url()` 负责。 |

---

## 路径

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `HARNESSLENS_ROOT` | 包所在目录的上一级 | 仓库根。只有把包装到仓库外时才需要设置。 |
| `HAI_REPO_ROOT` | 同上 | MCP server 子进程用的仓库根，由父进程自动注入。 |
| `OPENCODE_PREFIX` | `~/.opencode` | opencode 安装前缀；实际用的是 `<prefix>/bin/opencode`。 |
| `HAI_OPENCODE_BIN` | — | 直接指定 opencode 可执行文件，优先级高于 `OPENCODE_PREFIX`。 |
| `PI_AGENT_BIN` / `PI_BIN` | `<repo>/.pi-agent/node_modules/.bin/pi` | pi 可执行文件。 |
| `HAI_CODEX_MODELS_CACHE` | `~/.codex/models_cache.json` | codex 模型目录缓存，会被复制进每个隔离的 `CODEX_HOME`。 |
| `HAI_TERMINAL_BENCH_CACHE` | 工作区内 `.cache/terminal_bench/shared_trials` | terminal-bench 共享 trial 缓存。 |
| `TAU2_DATA_DIR` | `third_party/tau3-bench/data` | tau2 数据目录。checkout 不在约定位置时用它指过去。 |
| `TB_DOCKER` | `PATH` 上的 `docker` | docker 可执行文件路径。 |

查找可执行文件时，**`PATH` 上已有的永远优先**于上面这些变量。

---

## 预算与并发

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `HAI_TOTAL_CREATION_BUDGET` | `200` | 一次 run 允许创建的隔离智能会话总数（硬上限）。 |
| `HAI_TRAIN_ROLLOUT_REPEATS` | `2` | 每个 TRAIN 任务的 trial 数，只允许 `1` 或 `2`。论文实验用 2。 |
| `HAI_MAX_ROLLOUT_CONCURRENCY` | `20` | rollout 并发，允许 1–30。 |
| `HAI_MAX_ANALYSIS_CONCURRENCY` | `20` | 分析类角色的并发。 |
| `HAI_PROVIDER_MAX_CONCURRENCY` | `20` | provider 代理层的全局并发闸。 |
| `HAI_PROVIDER_TRIAL_MAX_CONCURRENCY` | `20` | 单个 trial 维度的 provider 并发闸。 |
| `HAI_PROVIDER_RETRY_ATTEMPTS` | `4` | provider 失败重试次数。 |
| `HAI_PROVIDER_SLOT_DIR` | `/tmp/harnesslens-provider-slots` | 跨进程并发槽的锁目录。 |
| `HAI_PROVIDER_TRIAL_SLOT_DIR` | `/tmp/harnesslens-provider-trial-slots` | 同上，trial 维度。 |
| `HAI_OPENCODE_TURN_RETRY_ATTEMPTS` | `0` | opencode 单轮重试次数。 |
| `HAI_MIN_MEM_GB` | `0`（不检查） | rollout 前要求的最小空闲内存 GB。 |
| `HAI_MIN_FREE_GB` | `0`（不检查） | rollout 前要求的最小空闲磁盘 GB。轨迹与工作区很占盘，长跑建议设。 |

## 迭代协议

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `HAI_CONFIRMATION_MODE` | `always` | `always` = 晋升前必须过 confirm 复验；`off` = 跳过。设成 `off` 跑出来的结果只能当探索性证据。 |
| `HAI_BASELINE_REUSE_POLICY` | — | 控制 baseline event 的复用严格程度。 |
| `HAI_TAU2_TIMEOUT_PER_TURN_S` | 见 `rollout_bridge.py` | tau2 单轮超时。 |
| `HAI_KEEP_TRAJECTORY_WORKSPACE` | `0` | 设为 `1` 保留每个 trial 的工作区（很占盘，调试用）。 |

---

## 分 cell 的变量

### tau2（retail / banking）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `HAI_TAU2_RETRIEVAL_CONFIG` | `bm25` | banking_knowledge 的检索配置，不能为空。 |
| `TAU2_MAX_TOOL_RESULT` | `0`（不截断） | 工具结果的最大字符数。 |
| `HAI_TAU2_LLM_MODEL` | — | 覆盖 tau2 用户模拟器所用模型。 |
| `HAI_TAU2_MCP_DEBUG_LOG` | — | tau2 MCP bridge 的调试日志路径。 |

### Terminal-Bench

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `HAI_DOCKER_ROOT` | `~/dockers` | rootless dockerd 的状态目录，**其余全部由它推导**。状态目录在别的盘上时必须设，否则守护进程会用一个空目录，表现为"镜像和容器全没了"。 |
| `HAI_DOCKER_HOST` | 由 `HAI_DOCKER_ROOT` 推导 | docker socket。显式设置时会校验它确实位于状态目录内，不一致直接报错。 |
| `DOCKER_HOST` | — | 标准 docker 变量，优先级最高。 |
| `TB_IMAGE_TEMPLATE` | 见 `terminal_images.py` | 任务镜像命名模板。 |
| `TB_NO_REBUILD` | — | 设为 `1` 时禁止重建镜像（要求已预拉取）。 |
| `TB_SHARED_NETWORK` | `harnesslens-terminal-bench` | 容器共享网络名。 |
| `TB_ENABLE_CONTAINER_CLASH` | 空 | 是否在容器内启用 clash 代理。 |
| `TB_CLASHCTL_HOME` | `~/clashctl` | clash 控制器目录（内含 `bin/mihomo`）。 |
| `TB_CONTAINER_PROXY_URL` / `TB_CONTAINER_HTTP_PROXY` / `TB_CONTAINER_HTTPS_PROXY` | — | 容器内代理地址。 |
| `TB_MODEL_ENDPOINT_NO_PROXY` | `1` | 让 provider 端点绕开代理。 |
| `TB_CLASH_START_CONCURRENCY` | `2` | clash 启动并发。 |
| `TB_OPENCODE_VERSION` | `latest` | 注入容器的 opencode 版本。 |
| `TB_SKIP_OPENCODE_INSTALL` | 空 | 跳过容器内 opencode 安装。 |
| `TB_OPENCODE_MODEL` | `deepseek/deepseek-v4-flash` | 容器内 opencode 使用的模型。 |
| `TB_CLASH_DNS_URL` | 空 | clash 的 DNS 上游地址。 |
| `TB_DOCKER_HOST_PROXY_HOST` | 自动探测 | 容器访问宿主代理时使用的地址。 |

`.venv/bin/python scripts/check_env.py --cell terminal-bench` 会打印它实际要连的 socket
及其来源（`DOCKER_HOST` 还是默认值），socket 不存在时报 MISS。


---

## 框架自动设置的变量（不要手动配）

下面这些由 `tau2_driver._configure_tau2_deepseek_env()` 从 `DEEPSEEK_*` 推导后写入
子进程环境，供 tau2 内部的 litellm 使用。手动设置它们不会生效，只会造成困惑：

| 变量 | 来源 |
| --- | --- |
| `OPENAI_API_KEY` | `DEEPSEEK_API_KEY` |
| `OPENAI_BASE_URL` / `OPENAI_API_BASE` | `config.provider_base_url()` 的解析结果 |
| `HAI_REPO_ROOT` | 父进程的仓库根，注入给 MCP server 子进程 |

要改端点或密钥，改 `DEEPSEEK_BASE_URL` / `DEEPSEEK_API_KEY`。

---

## 运行时工作区根目录

下面这几个变量指定各 harness 在 rollout 时的 cwd 根。默认在 run 产物树内，
只有当那块盘太小或不是本地盘时才需要改：

`HAI_OPENCODE_RUNTIME_ROOT`、`HAI_CODEX_RUNTIME_CWD_ROOT`、
`HAI_PI_RUNTIME_CWD_ROOT`

---

## 验证配置

```bash
.venv/bin/python scripts/check_env.py --cell retail --harness opencode
```

它会真的去解析 `benchmark_config()`，所以报出来的是所选 cell **实际**需要的
文件、虚拟环境和可执行文件，而不是一份猜测清单。缺任何一项都会以非 0 退出。
