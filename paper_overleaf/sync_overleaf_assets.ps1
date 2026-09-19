param(
    [switch]$Init
)

$ErrorActionPreference = "Stop"

$OverleafDir = $PSScriptRoot
$RepoRoot = Split-Path -Parent $OverleafDir
$TablesDir = Join-Path $OverleafDir "tables"
$FiguresDir = Join-Path $OverleafDir "figures"
$Copied = New-Object System.Collections.Generic.List[string]

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Text
    )
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($Path, $Text, $utf8)
}

function Copy-RequiredFile {
    param(
        [Parameter(Mandatory = $true)][string]$RelativeSource,
        [Parameter(Mandatory = $true)][string]$RelativeDestination
    )
    $srcPath = Join-Path $RepoRoot $RelativeSource
    $destPath = Join-Path $OverleafDir $RelativeDestination
    if (-not (Test-Path -LiteralPath $srcPath)) {
        throw "Missing source file: $srcPath"
    }
    $destParent = Split-Path -Parent $destPath
    if (-not (Test-Path -LiteralPath $destParent)) {
        New-Item -ItemType Directory -Force -Path $destParent | Out-Null
    }
    Copy-Item -LiteralPath $srcPath -Destination $destPath -Force
    $Copied.Add($RelativeDestination) | Out-Null
}

New-Item -ItemType Directory -Force -Path $TablesDir | Out-Null
New-Item -ItemType Directory -Force -Path $FiguresDir | Out-Null

Copy-RequiredFile "Regression\outputs\paper_tables\descriptive_correlation_table.tex" "tables\descriptive_correlation_table.tex"
Copy-RequiredFile "Regression\outputs\paper_tables\descriptive_rows.tex" "tables\descriptive_rows.tex"
Copy-RequiredFile "Regression\outputs\paper_tables\correlation_rows.tex" "tables\correlation_rows.tex"
Copy-RequiredFile "Regression\outputs\paper_tables\baseline_regression_table_ims.tex" "tables\baseline_regression_table_ims.tex"
Copy-RequiredFile "Regression\outputs\paper_tables\baseline_regression_table_ims_d.tex" "tables\baseline_regression_table_ims_d.tex"
Copy-RequiredFile "validation\P1\Two way cluster\outputs\tables\robustness_regression_table.tex" "tables\robustness_regression_table.tex"
Copy-RequiredFile "validation\P1\GPS-IPW\results\tables\gps_binary_matching_table.tex" "tables\gps_binary_matching_table.tex"
Copy-RequiredFile "DeepGreen_validation_revision\figures\layer_a_validation.pdf" "figures\layer_a_validation.pdf"
Copy-RequiredFile "DeepGreen_validation_revision\figures\layer_b_validation.pdf" "figures\layer_b_validation.pdf"

$descTable = Join-Path $TablesDir "descriptive_correlation_table.tex"
$descText = [System.IO.File]::ReadAllText($descTable)
$descText = $descText.Replace(
    "\input{../Regression/outputs/paper_tables/descriptive_rows.tex}",
    "\input{tables/descriptive_rows.tex}"
)
$descText = $descText.Replace(
    "\input{../Regression/outputs/paper_tables/correlation_rows.tex}",
    "\input{tables/correlation_rows.tex}"
)
Write-Utf8NoBom -Path $descTable -Text $descText

if ($Init) {
    $paperDest = Join-Path $OverleafDir "Paper.tex"
    if (Test-Path -LiteralPath $paperDest) {
        throw "overleaf/Paper.tex already exists. Refusing to overwrite. Omit -Init to sync tables/figures only."
    }
    Copy-RequiredFile "Writting_paper\Paper.tex" "Paper.tex"
    $paperText = [System.IO.File]::ReadAllText($paperDest)
    $replacements = @{
        "\input{../Regression/outputs/paper_tables/descriptive_correlation_table.tex}" = "\input{tables/descriptive_correlation_table.tex}"
        "\input{../Regression/outputs/paper_tables/baseline_regression_table_ims.tex}" = "\input{tables/baseline_regression_table_ims.tex}"
        "\input{../Regression/outputs/paper_tables/baseline_regression_table_ims_d.tex}" = "\input{tables/baseline_regression_table_ims_d.tex}"
        "\input{../validation/P1/Two way cluster/outputs/tables/robustness_regression_table.tex}" = "\input{tables/robustness_regression_table.tex}"
        "\input{../validation/P1/GPS-IPW/results/tables/gps_binary_matching_table.tex}" = "\input{tables/gps_binary_matching_table.tex}"
        "\includegraphics[width=0.85\textwidth]{../DeepGreen_validation_revision/figures/layer_a_validation.pdf}" = "\includegraphics[width=0.85\textwidth]{figures/layer_a_validation.pdf}"
        "\includegraphics[width=0.85\textwidth]{../DeepGreen_validation_revision/figures/layer_b_validation.pdf}" = "\includegraphics[width=0.85\textwidth]{figures/layer_b_validation.pdf}"
    }
    foreach ($old in $replacements.Keys) {
        if ($paperText.IndexOf($old) -lt 0) {
            throw "Expected path not found in Paper.tex: $old"
        }
        $paperText = $paperText.Replace($old, $replacements[$old])
    }
    Write-Utf8NoBom -Path $paperDest -Text $paperText
}

Write-Host "Synced $($Copied.Count) file(s) into overleaf/:"
$Copied | ForEach-Object { Write-Host "  $_" }
if ($Init) {
    Write-Host "Initialized overleaf/Paper.tex with Overleaf-relative paths."
}
else {
    Write-Host "Left overleaf/Paper.tex unchanged."
}
