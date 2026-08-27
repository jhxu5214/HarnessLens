# HarnessLens

**Verify Smarter, Evolve Further：通过行为感知验证实现高效 Harness 演化**

本仓库提供论文在 OpenCode、Codex CLI 和 Pi Coding Agent 上的实现，包括完整的 TRAIN
演化流程与独立的 blind TEST 评测入口。

[English](README.md)

## 目录结构

```text
harnesslens/
  core/               配置、预算、持久化与共享协议
  evolution/          Controller 及诊断/演化阶段
  harnesses/          可编辑面、manifest 与 agent runtime adapter
  benchmarks/         benchmark 任务驱动、MCP server 与评分逻辑
  evaluation/         rollout service 与 blind TEST 评测
  infrastructure/     进程、代理、provider 与容器工具
run_*.py              完整回路及 blind TEST 的公开入口
configs/              benchmark runtime 配置
assets/               vendored harness 文档与 canonical split 元数据
scripts/              安装、环境体检、单阶段工具和维护脚本
tests/                离线测试及 3 组 live runtime 测试
third_party/          外部 benchmark checkout（不随仓库分发）
runs/                 run 产物（已 gitignore）
```

## 依赖

| | |
| --- | --- |
| Python | 3.11+ |
| Python 包 | `pyyaml`、`httpx`、`loguru` 和 `pytest` |
| Provider | OpenAI-compatible chat-completions endpoint |
| Agent runtime | **opencode**、**codex**、**pi** 至少一个 |
| Benchmark | [docs/benchmarks_zh.md](docs/benchmarks_zh.md) 中列出的 checkout，放到或软链到 `third_party/` |
| 仅 Terminal-Bench | 可用的 Docker host，每个任务使用一个容器 |

论文实验的 target-agent 和 evolution roles 均使用 `deepseek-v4-flash-preview`，harness
版本分别为 OpenCode v1.17.13、Codex CLI v0.144.4 和 Pi Coding Agent v0.80.10。
Benchmark 数据和 agent runtime 体积较大且许可独立，因此不随仓库分发；复现文档固定了
所需版本和配置。

## 安装

```bash
git clone https://github.com/jhxu5214/HarnessLens.git
cd HarnessLens

# 创建 .venv、安装 HarnessLens 依赖，并生成 .env。
scripts/setup.sh

# 配置 provider。
$EDITOR .env  # DEEPSEEK_API_KEY；DEEPSEEK_BASE_URL 可选
```

## 环境准备

Benchmark 数据和 agent runtime 都是外部依赖。请按照
[docs/benchmarks_zh.md](docs/benchmarks_zh.md) clone 到固定 revision，或把已有 checkout
软链到 `third_party/`。机器相关的路径、Docker 和代理设置见
[docs/configuration_zh.md](docs/configuration_zh.md)。

| 环境 | 额外本地依赖 |
| --- | --- |
| Retail / Banking Knowledge | tau2-bench checkout 及其虚拟环境；Banking 还需要知识库语料 |
| Terminal-Bench 2.0 | Terminal-Bench checkout 及可连接的 Docker host |
| BIRD Mini-Dev | BIRD checkout、prompt JSONL 及 SQLite 数据库 |

## 快速开始

```bash
.venv/bin/python scripts/check_env.py --cell retail --harness opencode
scripts/run_e2e.sh --run-id retail-001 --cell retail --harness opencode
```

通过 `--cell` 和 `--harness` 选择其他支持的环境或 harness。中断后用相同 `--run-id`
重新执行即可恢复。产物位于 `runs/train/<run-id>/`；最终 harness 是
`submission/final.json`，`controller_state.json` 记录 checkpoint。

### Blind TEST 评测

Blind TEST 与演化流程刻意分离。TRAIN Controller 产出 `submission/final.json` 后才能运行：

```bash
.venv/bin/python run_test_candidate.py --benchmark retail --harness opencode \
  --run-id retail-001-test \
  --patch-json runs/train/retail-001/submission/final.json
```

其他环境的 `--benchmark` 分别使用 `banking`、`terminal-bench` 和
`bird-mini-dev-challenging`。`run_test_baseline.py` 使用相同 blind protocol 评测对应的
未修改 harness。

## 测试

```bash
scripts/run_tests.sh
```

传入 `--live` 可额外运行真实 runtime 和 provider 检查。

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/architecture_zh.md](docs/architecture_zh.md) | 模块、验证协议和产物布局 |
| [docs/configuration_zh.md](docs/configuration_zh.md) | 环境变量及 runtime 行为 |
| [docs/benchmarks_zh.md](docs/benchmarks_zh.md) | benchmark checkout、pinned revision 和 cell 配置 |
| [docs/troubleshooting_zh.md](docs/troubleshooting_zh.md) | 常见错误及诊断方法 |

## 适用范围与局限

论文实验覆盖一个模型家族、三种 harness 和四个公开 benchmark。Interaction unit 使预算
可审计，但没有对不同角色和环境的 token、延迟或货币成本做归一化。行为感知演化在多个
任务暴露同一种可控行为时最有效；面对目标和执行路径高度多样的任务集，孤立修复可能缺少
可迁移证据，此时正确结果可能是不修改 incumbent。

## 许可证

HarnessLens 使用 [MIT License](LICENSE)。`assets/` 下 vendored 的文档沿用上游条款；
benchmark 数据和 agent runtime 不随仓库分发。
