#!/bin/bash
# SessionStart hook for Claude Code cloud sessions: restart a long job after the sandbox VM was
# paused and brought back. Files survive that; processes do not.
#
# Opt-in per job: the job leaves a marker file (default ~/.momij-resume) whose first line is the
# path of an IDEMPOTENT resume script, and removes the marker when it finishes. No marker, no
# remote session, or another start source (clear/compact) -> this hook does nothing.
set -euo pipefail

[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0
marker="${MOMIJ_RESUME_MARKER:-$HOME/.momij-resume}"
[ -f "$marker" ] || exit 0

# stdin is the hook input JSON; only `startup` and `resume` mean "fresh VM".
source_kind=$(sed -n 's/.*"source"[[:space:]]*:[[:space:]]*"\([a-z]*\)".*/\1/p' | head -1 || true)
case "$source_kind" in startup | resume) ;; *) exit 0 ;; esac

script=$(head -1 "$marker")
if [ ! -f "$script" ]; then
  echo "[resume-experiment] $marker points to a missing script: $script"
  exit 0
fi
if pgrep -f -- "$script" >/dev/null; then
  echo "[resume-experiment] $script is already running; nothing to do."
  exit 0
fi

log="${script%.sh}.resume.log"
setsid nohup sh "$script" >"$log" 2>&1 </dev/null &
echo "[resume-experiment] the sandbox was paused and its processes are gone; restarted $script in the background (log: $log)."
