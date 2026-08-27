#!/usr/bin/env python3
"""Remove Terminal-Bench task containers left behind by a killed run.

Containers are torn down in a ``finally`` block, which ``SIGKILL`` skips — and
a container is not in the orchestrator's process tree, so nothing else reclaims
it. Each one is stamped with the pid and boot that created it, so this only
removes containers whose run is provably gone. Live runs are untouched, and
images are never removed.

    python scripts/reap_containers.py            # show what would go
    python scripts/reap_containers.py --remove   # actually remove it

`run_terminal_batch` already reclaims on start; this is for cleaning up without
starting a run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harnesslens.core.config import load_repo_env, repo_root  # noqa: E402
from harnesslens.infrastructure.container_reaper import (  # noqa: E402
    orphaned_containers,
    reap_orphaned_containers,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--remove",
        action="store_true",
        help="remove them; without this the orphans are only listed",
    )
    args = parser.parse_args()
    load_repo_env(repo_root())

    orphans = orphaned_containers()
    if not orphans:
        print("No orphaned containers.")
        return 0

    for orphan in orphans:
        print(f"  {orphan['id']}  {orphan['name']}  ({orphan['reason']})")

    if not args.remove:
        print(f"\n{len(orphans)} orphaned container(s). Pass --remove to delete them.")
        return 0

    removed = reap_orphaned_containers()
    print(f"\nRemoved {len(removed)} of {len(orphans)} container(s). Images untouched.")
    return 0 if len(removed) == len(orphans) else 1


if __name__ == "__main__":
    raise SystemExit(main())
