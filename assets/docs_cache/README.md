# Vendored harness documentation cache

The Harness Query stage feeds official harness documentation to the analyst
models as `local_documentation` evidence. Fetching it live would make a run
depend on the network and on whatever the upstream docs say that day, so the
exact snapshots used by the reported experiments are frozen here.

| Directory | Upstream | Snapshot |
| --- | --- | --- |
| `opencode/` | <https://opencode.ai/docs/> (see `opencode/SOURCE.txt`) | opencode CLI 1.14.27, fetched 2026-06-06 |
| `codex/` | <https://developers.openai.com/codex/config-reference> | fetched 2026-06-09 |

Pi documentation is *not* vendored: it ships inside the `pi-coding-agent` npm
package and is read from
`<repo>/.pi-agent/node_modules/@earendil-works/pi-coding-agent/docs/`.
Install the pi runtime (see the README) to make that evidence available.

To refresh a snapshot, re-fetch the pages listed in the corresponding
`SOURCE.txt` and record the new version and date there.
