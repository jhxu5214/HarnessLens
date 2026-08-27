from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

import yaml


_APT_LOCK_HELPERS = r"""
wait_for_apt_locks() {
  clear=0
  for attempt in $(seq 1 180); do
    busy=0
    for comm in /proc/[0-9]*/comm; do
      [ -r "$comm" ] || continue
      read -r name < "$comm" || continue
      case "$(cat "$comm" 2>/dev/null)" in apt|apt-get|dpkg) busy=1; break ;; esac
    done
    if [ "$busy" -eq 0 ]; then clear=$((clear + 1)); [ "$clear" -ge 3 ] && return 0; else clear=0; fi
    sleep 2
  done
}
apt_retry() {
  wait_for_apt_locks
  for attempt in $(seq 1 60); do
    output="$(apt-get -o DPkg::Lock::Timeout=300 "$@" 2>&1)" && { printf '%s\n' "$output"; return 0; }
    status=$?
    printf '%s\n' "$output"
    case "$output" in *"Could not get lock"*|*"Unable to acquire the dpkg frontend lock"*) sleep 3 ;; *) return "$status" ;; esac
  done
  return 1
}
""".strip()


def install_terminal_runtime_hooks(tb: Any) -> None:
    """Patch only the worker-local legacy adapter hooks used by the HarnessLens launcher.

    The surrounding trajectory contract remains unchanged, while the HarnessLens-owned
    functions make task-container initialization portable to minimal images and
    prevent Compose from allocating one network per trial.
    """

    def compose_override(tmp: Path) -> Path:
        proxy_env = tb._container_proxy_env()
        container_env = {**proxy_env, "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", "")}
        override = {
            "services": {
                "client": {
                    "extra_hosts": ["host.docker.internal:host-gateway"],
                    "environment": container_env,
                }
            },
            "networks": {
                "default": {
                    "external": True,
                    "name": os.environ.get("TB_SHARED_NETWORK", "harnesslens-terminal-bench"),
                }
            },
        }
        path = Path(tmp) / "docker-compose.override.harness.yaml"
        path.write_text(yaml.safe_dump(override, sort_keys=True))
        return path

    def copy_host_node_tools(cid: str) -> bool:
        root = tb._host_node_root()
        if root is None:
            return False
        prep = tb._dexec_result(
            cid,
            "rm -rf /opt/harness-node && mkdir -p /opt/harness-node/bin /opt/harness-node/lib/node_modules",
            timeout=60,
            workdir="/",
        )
        if prep.returncode != 0:
            return False
        tb._dcp_in(root / "bin" / "node", cid, "/opt/harness-node/bin/node")
        tb._dcp_in(root / "lib" / "node_modules" / "npm", cid, "/opt/harness-node/lib/node_modules/npm")
        script = f"""
set -e
export DEBIAN_FRONTEND=noninteractive
{_APT_LOCK_HELPERS}
apt_retry update
apt_retry install -y libatomic1
chmod +x /opt/harness-node/bin/node
ln -sf ../lib/node_modules/npm/bin/npm-cli.js /opt/harness-node/bin/npm
ln -sf ../lib/node_modules/npm/bin/npx-cli.js /opt/harness-node/bin/npx
ln -sf /opt/harness-node/bin/node /usr/local/bin/node
ln -sf /opt/harness-node/bin/npm /usr/local/bin/npm
ln -sf /opt/harness-node/bin/npx /usr/local/bin/npx
node --version
npm --version
""".strip()
        return tb._dexec_result(cid, script, timeout=600, workdir="/").returncode == 0

    def ensure_opencode(cid: str, timeout: int = 1200) -> str:
        check = tb._dexec_result(cid, "command -v opencode && opencode --version", timeout=60)
        if check.returncode == 0:
            return (check.stdout or b"").decode("utf-8", "replace")
        if os.environ.get("TB_SKIP_OPENCODE_INSTALL", "").lower() in {"1", "true", "yes", "on"}:
            raise RuntimeError("opencode is not installed in the task container")
        if tb._dexec_result(cid, "command -v npm", timeout=60, workdir="/").returncode != 0:
            copy_host_node_tools(cid)
        version = shlex.quote(os.environ.get("TB_OPENCODE_VERSION", "latest"))
        script = f"""
set -e
export DEBIAN_FRONTEND=noninteractive
{_APT_LOCK_HELPERS}
if ! command -v curl >/dev/null 2>&1; then
  apt_retry update
  apt_retry install -y curl ca-certificates
fi
if [ -d /opt/harness-node ]; then
  apt_retry update
  apt_retry install -y libatomic1
  export PATH="/opt/harness-node/bin:$PATH"
fi
if ! command -v npm >/dev/null 2>&1; then
  apt_retry update
  apt_retry install -y nodejs npm
fi
npm i -g opencode-ai@{version}
prefix="$(npm prefix -g 2>/dev/null || true)"
if [ -n "$prefix" ] && [ -e "$prefix/bin/opencode" ]; then ln -sf "$prefix/bin/opencode" /usr/local/bin/opencode; fi
if [ -e /opt/harness-node/bin/opencode ]; then ln -sf /opt/harness-node/bin/opencode /usr/local/bin/opencode; fi
command -v opencode
opencode --version
""".strip()
        return tb._dexec_detached_script(cid, script, timeout=timeout, workdir="/app")

    tb._compose_override = compose_override
    tb._copy_host_node_tools_to_container = copy_host_node_tools
    tb._ensure_opencode_in_container = ensure_opencode
