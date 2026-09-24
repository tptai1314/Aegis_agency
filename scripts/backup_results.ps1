# Pull a run bundle (results + provenance + verdict cache) from a GPU server to this machine.
# Run on the Windows dev machine, from the repository root.
#
#   1) On the server:  bash scripts/ec2/backup_results.sh --outputs outputs/real_harmbench
#   2) On Windows:
#        powershell -ExecutionPolicy Bypass -File scripts/backup_results.ps1 `
#            -HostTarget user@host -Key "$env:USERPROFILE\.ssh\id_ed25519" `
#            -RemoteBundle "~/aegis_backups/aegis_real_harmbench_20260101T000000Z.tar.gz"
#
# It downloads the bundle plus its .sha256 sidecar, verifies the checksum locally, unpacks into
# .\backups\<bundle name>, and prints the finalize_run.py command to run next.
param(
  [Parameter(Mandatory = $true)][string]$HostTarget,
  [string]$Key = "$env:USERPROFILE\.ssh\id_ed25519",
  [Parameter(Mandatory = $true)][string]$RemoteBundle,
  [string]$LocalDest = "backups",
  [switch]$SkipVerify
)

$ErrorActionPreference = "Stop"

$name = [System.IO.Path]::GetFileName($RemoteBundle)
if (-not $name.EndsWith(".tar.gz")) { throw "RemoteBundle must be a .tar.gz produced by backup_results.sh" }

$dest = Join-Path (Get-Location) $LocalDest
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$localBundle = Join-Path $dest $name

Write-Host "==> Downloading $RemoteBundle"
scp -i $Key "${HostTarget}:${RemoteBundle}" $localBundle
if ($LASTEXITCODE -ne 0) { throw "scp of the bundle failed" }

$haveSidecar = $true
try {
  scp -i $Key "${HostTarget}:${RemoteBundle}.sha256" "$localBundle.sha256"
  if ($LASTEXITCODE -ne 0) { $haveSidecar = $false }
} catch { $haveSidecar = $false }

if ($haveSidecar -and -not $SkipVerify) {
  Write-Host "==> Verifying sha256"
  $expected = ((Get-Content "$localBundle.sha256" -Raw).Trim() -split '\s+')[0]
  $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $localBundle).Hash.ToLower()
  if ($expected.ToLower() -ne $actual) {
    throw "checksum mismatch: expected $expected, got $actual. Do not trust this bundle."
  }
  Write-Host "    OK ($actual)"
} else {
  Write-Warning "No .sha256 sidecar verified; integrity of the transfer is unconfirmed."
}

$unpacked = Join-Path $dest ($name -replace '\.tar\.gz$','')
New-Item -ItemType Directory -Force -Path $unpacked | Out-Null
Write-Host "==> Unpacking to $unpacked"
tar -xzf $localBundle -C $unpacked
if ($LASTEXITCODE -ne 0) { throw "tar extraction failed" }

# The server-side run directory is the relative path stored inside the bundle.
$runDir = Get-ChildItem -Path $unpacked -Recurse -File -Filter "real_evaluation_provenance.json" |
          Select-Object -First 1
if ($runDir) {
  $runPath = $runDir.Directory.FullName
  Write-Host "==> Run directory: $runPath"
  Write-Host ""
  Write-Host "Next:"
  Write-Host "    python scripts/check_ready.py --config configs/ec2_real_evaluation.yaml --offline"
  Write-Host "    python scripts/finalize_run.py --run `"$runPath`""
  Write-Host "    # after an independent repeat of the same frozen config:"
  Write-Host "    python scripts/finalize_run.py --run `"$runPath`" --replicated-by <second run> --set-paper-result"
} else {
  Write-Warning "No real_evaluation_provenance.json found inside the bundle; inspect $unpacked manually."
}
