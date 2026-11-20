$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$html = Get-Content -LiteralPath (Join-Path $root 'index.html') -Raw
$css = (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'out.css') -Raw) -replace '\\2c ', ',' -replace '\\', ''

$classes = New-Object System.Collections.Generic.List[string]
foreach ($m in [regex]::Matches($html, 'class="([^"]*)"')) {
    foreach ($t in ($m.Groups[1].Value -split '\s+')) { if ($t) { $classes.Add($t) } }
}
# Classes only produced by JS string literals / template concatenation.
@(
    # setHeader() tone dots
    'bg-emerald-400/70', 'bg-emerald-400', 'bg-sky-400/70', 'bg-sky-400',
    'bg-amber-400/70', 'bg-amber-400', 'bg-rose-400/70', 'bg-rose-400',
    # setBadge() variants
    'border-white/10', 'bg-white/[0.03]', 'text-slate-400',
    'border-sky-400/30', 'bg-sky-400/10', 'text-sky-200',
    'border-emerald-400/30', 'bg-emerald-400/10', 'text-emerald-200',
    'border-rose-400/30', 'bg-rose-400/10', 'text-rose-200',
    # renderBilling() plan badge
    'border-emerald-400/25', 'border-sky-400/25',
    # renderTrialMeter() progress bar + label
    'bg-rose-500', 'bg-amber-400', 'text-rose-300', 'text-amber-300', 'text-sky-300'
) | ForEach-Object { $classes.Add($_) }

# Custom / non-utility hooks that are styled by the hand-written CSS, not Tailwind.
$custom = @('panel', 'drop', 'step', 'dot', 'label', 'core', 'halo',
    'success-panel', 'success-glow', 'success-circle', 'success-tick', "'", '+')

$missing = New-Object System.Collections.Generic.List[string]
foreach ($c in ($classes | Sort-Object -Unique)) {
    if ($custom -contains $c) { continue }
    $pattern = '(?<![\w-])\.' + [regex]::Escape($c) + '(?=[,{:\s>])'
    if (-not [regex]::IsMatch($css, $pattern)) { $missing.Add($c) }
}

"unique classes checked : " + (($classes | Sort-Object -Unique).Count)
"missing from out.css   : " + $missing.Count
$missing | ForEach-Object { "  - $_" }
