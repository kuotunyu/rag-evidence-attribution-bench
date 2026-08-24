# Verified Annotator Kit Runbook

This runbook applies only after the owner separately authorizes a human pilot and the
coordinator delivers one verified A or B kit. B0.1 engineering alone does not authorize
annotation. Do not obtain the other annotator's kit, the coordinator manifest, repository
access, another person's state, or any prior decision.

The coordinator may stage a kit only after the exact source candidate has produced four
byte-identical, Git-blob-bound `0.2.0` wheels and one passed
`platform-verification-receipt-v2` on both Windows and Linux. These platform receipts, the root
handoff receipt, smoke evidence, and the coordinator manifest remain coordinator-only and must not
be delivered to either annotator. A GitHub Actions result is a review gate, not signed provenance.

## Fixed privacy and stop rules

- Work only from the visible question and passages. Do not search the web or repository.
- Do not enter a name, email address, employer, private filesystem path, or other PII.
- Keep the kit, state, exports, and return checksums private and outside any Git checkout.
- Use only `127.0.0.1`. Stop if the browser or tool requests network access.
- The online package-index step ends before state creation. From launcher start through export,
  follow the documented loopback-only local workflow; this is not an air-gap or OS sandbox claim.
- Stop immediately if `SHA256SUMS` fails, the wheel/package is missing, the package shows
  another pseudonym, the instruction version is not `pilot-v0.2.2-draft`, or the displayed
  instruction hash differs from the package hash.
- Never continue after a hash mismatch. Preserve the files unchanged and notify the
  coordinator through the separately agreed private channel.

Each kit must contain exactly one `ann-pilot-a.json` or `ann-pilot-b.json`, one wheel,
`annotation-requirements-py311.lock`, `ONBOARDING.md`, this runbook, `start.ps1`, `start.sh`,
and `SHA256SUMS`. It must not contain the other package or any file with `manifest` in its name.

## Windows PowerShell 7+

Open PowerShell in the kit directory. Verify every delivered file before installation:

```powershell
$ErrorActionPreference = "Stop"
$KitRoot = (Get-Location).Path
Get-Content -LiteralPath (Join-Path $KitRoot "SHA256SUMS") | ForEach-Object {
    if ($_ -notmatch '^([0-9a-f]{64})  (.+)$') { throw "Malformed SHA256SUMS line." }
    $Expected = $Matches[1]
    $Relative = $Matches[2]
    $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $KitRoot $Relative)).Hash.ToLowerInvariant()
    if ($Actual -ne $Expected) { throw "SHA-256 mismatch: $Relative" }
}
```

Create a Python 3.11 environment outside the kit. Bootstrap only the locked binary runtime,
then install the one verified wheel without dependency resolution. Do not upgrade pip, remove
`--only-binary=:all:`, resolve a wheel extra, or permit an sdist/build-isolation fallback:

```powershell
$PrivateRoot = Join-Path $env:LOCALAPPDATA "reab-pilot-private"
$VenvRoot = Join-Path $PrivateRoot "venv"
$StateRoot = Join-Path $PrivateRoot "state"
$ReturnRoot = Join-Path $PrivateRoot "return"
$null = New-Item -ItemType Directory -Force -Path $PrivateRoot
py -3.11 -m venv $VenvRoot
$Python = Join-Path $VenvRoot "Scripts/python.exe"
$Wheels = @(Get-ChildItem -LiteralPath $KitRoot -Filter *.whl -File)
if ($Wheels.Count -ne 1) { throw "Kit must contain exactly one wheel." }
& $Python -m pip install --require-hashes --only-binary=:all: -r (Join-Path $KitRoot "annotation-requirements-py311.lock")
& $Python -m pip install --no-deps "$($Wheels[0].FullName)"
& $Python -c "import sys; assert sys.version_info[:2] == (3, 11)"
& $Python -m rag_evidence annotation --help
```

Activate the environment and launch the kit-specific script:

```powershell
& (Join-Path $VenvRoot "Scripts/Activate.ps1")
& (Join-Path $KitRoot "start.ps1") -StateRoot $StateRoot -Port 8001
```

After all tasks are submitted, keep the server running and export both streams in a second
PowerShell window:

```powershell
$null = New-Item -ItemType Directory -Force -Path $ReturnRoot
Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8001/api/export/submissions.jsonl" -OutFile (Join-Path $ReturnRoot "submissions.jsonl")
Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8001/api/export/amendments.jsonl" -OutFile (Join-Path $ReturnRoot "amendments.jsonl")
Get-ChildItem -LiteralPath $ReturnRoot -File | Where-Object Name -ne "SHA256SUMS" | Sort-Object Name | ForEach-Object {
    "{0}  {1}" -f (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant(), $_.Name
} | Set-Content -Encoding utf8NoBOM -LiteralPath (Join-Path $ReturnRoot "SHA256SUMS")
```

Return only `submissions.jsonl`, the present `amendments.jsonl` (empty is valid), and the
return `SHA256SUMS`. Do not return drafts, state, the wheel, or the kit.

## POSIX shell on Linux

Open a shell in the kit directory and verify all delivered files:

```sh
set -eu
KIT_ROOT=$(pwd -P)
sha256sum -c SHA256SUMS
```

Create a Python 3.11 environment outside the kit. Bootstrap only the locked binary runtime,
then install the verified wheel without dependency resolution:

```sh
PRIVATE_ROOT=${XDG_STATE_HOME:-"$HOME/.local/state"}/reab-pilot-private
VENV_ROOT=$PRIVATE_ROOT/venv
STATE_ROOT=$PRIVATE_ROOT/state
RETURN_ROOT=$PRIVATE_ROOT/return
mkdir -p -- "$PRIVATE_ROOT"
python3.11 -m venv "$VENV_ROOT"
PYTHON=$VENV_ROOT/bin/python
set -- "$KIT_ROOT"/*.whl
[ "$#" -eq 1 ] || { echo "Kit must contain exactly one wheel." >&2; exit 2; }
"$PYTHON" -m pip install --require-hashes --only-binary=:all: -r "$KIT_ROOT/annotation-requirements-py311.lock"
"$PYTHON" -m pip install --no-deps "$1"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11)'
"$PYTHON" -m rag_evidence annotation --help
```

Activate the environment and launch the kit-specific script:

```sh
. "$VENV_ROOT/bin/activate"
chmod u+x "$KIT_ROOT/start.sh"
"$KIT_ROOT/start.sh" "$STATE_ROOT" 8001
```

After all tasks are submitted, keep the server running and export in a second shell:

```sh
mkdir -p -- "$RETURN_ROOT"
curl --fail --silent --show-error "http://127.0.0.1:8001/api/export/submissions.jsonl" --output "$RETURN_ROOT/submissions.jsonl"
curl --fail --silent --show-error "http://127.0.0.1:8001/api/export/amendments.jsonl" --output "$RETURN_ROOT/amendments.jsonl"
(cd "$RETURN_ROOT" && sha256sum submissions.jsonl amendments.jsonl > SHA256SUMS)
```

Return only `submissions.jsonl`, the present `amendments.jsonl` (empty is valid), and the
return `SHA256SUMS`. Keep the private state until the coordinator confirms checksum receipt;
then follow the owner's retention decision.
