param(
    [Parameter(Mandatory = $true)]
    [string]$StateRoot,
    [int]$Port = 8001
)

$ErrorActionPreference = "Stop"
$KitRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $MyInvocation.MyCommand.Path)).Path
$ResolvedState = [IO.Path]::GetFullPath($StateRoot)
$KitPrefix = $KitRoot.TrimEnd("\", "/") + [IO.Path]::DirectorySeparatorChar
if ($ResolvedState -eq $KitRoot -or $ResolvedState.StartsWith($KitPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "StateRoot must remain outside the delivery kit."
}
$null = New-Item -ItemType Directory -Force -Path $ResolvedState
$Package = Join-Path $KitRoot "ann-pilot-b.json"
python -m rag_evidence annotation serve --package $Package --state $ResolvedState --host 127.0.0.1 --port $Port
