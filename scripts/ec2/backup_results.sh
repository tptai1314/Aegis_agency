#!/usr/bin/env bash
# Pack a finished run (results + provenance + verdict caches) for transfer off the server.
# Run from the repository root ON THE SERVER:
#
#   bash scripts/ec2/backup_results.sh                                  # all of outputs/
#   bash scripts/ec2/backup_results.sh --outputs outputs/real_harmbench
#   bash scripts/ec2/backup_results.sh --dest /mnt/backups
#
# The verdict cache is part of the run record: it is the raw judge measurement the CSVs were
# computed from, and it is what makes a re-analysis cheap. Do not delete it before backing up.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUTS="outputs"
DEST="$HOME/aegis_backups"
while [ $# -gt 0 ]; do
  case "$1" in
    --outputs) OUTPUTS="$2"; shift 2 ;;
    --dest)    DEST="$2";    shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ ! -e "$OUTPUTS" ]; then
  echo "Nothing to back up: $OUTPUTS does not exist." >&2
  exit 1
fi

# Refuse to ship a run that still claims to be unverified paper output, unless it explicitly does.
if [ -f "$OUTPUTS/real_evaluation_provenance.json" ]; then
  grep -q '"is_paper_result": true' "$OUTPUTS/real_evaluation_provenance.json" \
    && echo "NOTE: this run is flagged is_paper_result=true." \
    || echo "NOTE: this run is still is_paper_result=false (expected until finalize_run.py approves it)."
fi

mkdir -p "$DEST"
NAME="$(basename "$OUTPUTS")"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BUNDLE="$DEST/aegis_${NAME}_${STAMP}.tar.gz"

echo "==> Packing $OUTPUTS -> $BUNDLE"
tar -czf "$BUNDLE" \
  --exclude='*.pyc' --exclude='__pycache__' \
  "$OUTPUTS" \
  $( [ -d audits/run_records ] && echo audits/run_records ) \
  $( [ -f audits/result_integrity_audit.md ] && echo audits/result_integrity_audit.md )

# Manifest: every file in the bundle plus the code/environment identity.
MANIFEST="$DEST/aegis_${NAME}_${STAMP}.manifest.txt"
{
  echo "bundle: $BUNDLE"
  echo "created_utc: $STAMP"
  echo "git_commit: $(git rev-parse HEAD 2>/dev/null || echo not-a-git-repo)"
  echo "git_dirty_paths: $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  echo "python: $(python -V 2>&1 || echo unknown)"
  echo "--- sha256 ---"
  find "$OUTPUTS" audits/run_records -type f 2>/dev/null | sort | while read -r f; do
    sha256sum "$f"
  done
} > "$MANIFEST"

sha256sum "$BUNDLE" > "$BUNDLE.sha256"
SIZE="$(du -h "$BUNDLE" | cut -f1)"

cat <<EOF

==> Done: $BUNDLE ($SIZE)
    manifest: $MANIFEST
    checksum: $BUNDLE.sha256

Pull it to the Windows dev machine (run this on Windows, from the repo root):

    powershell -ExecutionPolicy Bypass -File scripts/backup_results.ps1 \`
        -HostTarget <user@host> -Key "\$env:USERPROFILE\\.ssh\\<key>" \`
        -RemoteBundle "$BUNDLE"

Then record the run in the audit table:

    python scripts/finalize_run.py --run <unpacked run directory> [--replicated-by <second run>]
EOF
