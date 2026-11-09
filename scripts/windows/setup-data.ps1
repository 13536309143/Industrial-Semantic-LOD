param([Parameter(Mandatory=$true)][string]$DataRoot)
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$externalRoot = (Resolve-Path -LiteralPath $DataRoot).Path
$links = [ordered]@{
    'data' = ''
    '_downloaded_resources' = 'datasets\scenes'
    'lod_analysis_outputs' = 'experiments\inference'
    'analysis_reports' = 'experiments\reports'
    'outputs' = 'experiments\outputs'
    'experiments\pointnext' = 'experiments\training\pointnext'
}
foreach ($entry in $links.GetEnumerator()) {
    $link = Join-Path $repoRoot $entry.Key
    $target = if ($entry.Value) { Join-Path $externalRoot $entry.Value } else { $externalRoot }
    if (-not (Test-Path -LiteralPath $target -PathType Container)) {
        throw "Missing data directory: $target. Prepare or restore it before linking."
    }
    $existing = Get-Item -LiteralPath $link -Force -ErrorAction SilentlyContinue
    if ($existing) {
        if ($existing.LinkType -eq 'Junction' -and [IO.Path]::GetFullPath($existing.Target[0]).TrimEnd('\') -eq [IO.Path]::GetFullPath($target).TrimEnd('\')) {
            Write-Output "Already linked: $link"
            continue
        }
        throw "Existing path is not the expected junction: $link. This script never removes or replaces data."
    }
    New-Item -ItemType Junction -Path $link -Target $target | Out-Null
    Write-Output "$link -> $target"
}
