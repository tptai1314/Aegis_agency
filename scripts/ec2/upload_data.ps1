# Upload the prepared benchmark CSVs (D:\Data\benchmarks) to an EC2 instance.
# Run on the Windows dev machine:
#   powershell -ExecutionPolicy Bypass -File scripts/ec2/upload_data.ps1 `
#       -HostTarget ubuntu@<ec2-host> -Key ~\.ssh\aegis.pem
#
# Packs the benchmark directory into one tar, pushes it with scp, and extracts
# it on the server at /data/benchmarks (the config's data.root layout).
param(
  [Parameter(Mandatory = $true)]
  [string]$HostTarget,                       # e.g. ubuntu@ec2-1-2-3-4.compute.amazonaws.com
  [string]$Key = "$env:USERPROFILE\.ssh\id_ed25519",
  [string]$DataRoot = "D:\Data\benchmarks",
  [string]$RemoteRoot = "/data/benchmarks",
  [switch]$SkipVerify
)

$ErrorActionPreference = "Stop"
$temp = Join-Path $env:TEMP "aegis_benchmarks.tar"

if (-not (Test-Path -LiteralPath $DataRoot)) {
  Write-Error "Local benchmark root not found: $DataRoot"
}

Write-Host "==> Packing $DataRoot"
tar -cf $temp -C $DataRoot .
if ($LASTEXITCODE -ne 0) { throw "tar failed" }
Write-Host "    -> $temp ($([math]::Round((Get-Item $temp).Length / 1MB, 1)) MB)"

Write-Host "==> Uploading to $HostTarget"
scp -i $Key $temp "${HostTarget}:/tmp/aegis_benchmarks.tar"
if ($LASTEXITCODE -ne 0) { throw "scp failed" }

Write-Host "==> Extracting on remote at $RemoteRoot"
$remoteCmd = "sudo mkdir -p $RemoteRoot && " +
             "sudo tar -xf /tmp/aegis_benchmarks.tar -C $RemoteRoot && " +
             "sudo chown -R `$USER:$`USER $RemoteRoot && " +
             "rm -f /tmp/aegis_benchmarks.tar"
ssh -i $Key $HostTarget $remoteCmd
if ($LASTEXITCODE -ne 0) { throw "remote extraction failed" }

Remove-Item -LiteralPath $temp -ErrorAction SilentlyContinue

if (-not $SkipVerify) {
  Write-Host "==> Verifying on remote"
  $verify = "source ~/.bashrc; ls -1 $RemoteRoot"
  ssh -i $Key $HostTarget $verify
}

Write-Host "Done. Benchmarks are at $RemoteRoot on $HostTarget (see docs/ec2_experiment_guide.md)."