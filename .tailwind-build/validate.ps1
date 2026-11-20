$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$indexPath = Join-Path $root 'index.html'
$html = Get-Content -LiteralPath $indexPath -Raw

$problems = New-Object System.Collections.Generic.List[string]

# 1. Balanced <style> / <script> tags (comments removed first)
$markup = [regex]::Replace($html, '(?s)<!--.*?-->', '')
$styleOpen = ([regex]::Matches($markup, '<style\b')).Count
$styleClose = ([regex]::Matches($markup, '</style>')).Count
$scriptOpen = ([regex]::Matches($markup, '<script\b')).Count
$scriptClose = ([regex]::Matches($markup, '</script>')).Count
"style tags : $styleOpen open / $styleClose close"
"script tags: $scriptOpen open / $scriptClose close"
if ($styleOpen -ne $styleClose) { $problems.Add('unbalanced <style> tags') }
if ($scriptOpen -ne $scriptClose) { $problems.Add('unbalanced <script> tags') }

# 2. No blocked third-party origins are actually loaded (comments ignored)
foreach ($m in [regex]::Matches($markup, '(?:src|href)\s*=\s*["''](https?://[^"'']+)["'']')) {
    $url = $m.Groups[1].Value
    if ($url -notlike '*posthog.com*') { $problems.Add("external asset still loaded: $url") }
}
"external src/href : " + ([regex]::Matches($markup, '(?:src|href)\s*=\s*["'']https?://')).Count

# 3. Every utility class in the markup resolves to a rule in the inlined CSS
$styleCss = ''
foreach ($m in [regex]::Matches($html, '(?s)<style>(.*?)</style>')) { $styleCss += $m.Groups[1].Value }
$styleCss = $styleCss -replace '\\2c ', ',' -replace '\\', ''

$classes = New-Object System.Collections.Generic.List[string]
foreach ($m in [regex]::Matches($html, 'class="([^"]*)"')) {
    foreach ($t in ($m.Groups[1].Value -split '\s+')) { if ($t) { $classes.Add($t) } }
}
foreach ($m in [regex]::Matches($html, "className = '([^']*)'")) {
    foreach ($t in ($m.Groups[1].Value -split '\s+')) { if ($t) { $classes.Add($t) } }
}
@(
    'bg-emerald-400/70', 'bg-emerald-400', 'bg-sky-400/70', 'bg-sky-400',
    'bg-amber-400/70', 'bg-amber-400', 'bg-rose-400/70', 'bg-rose-400',
    'border-white/10', 'bg-white/[0.03]', 'text-slate-400',
    'border-sky-400/30', 'bg-sky-400/10', 'text-sky-200',
    'border-emerald-400/30', 'bg-emerald-400/10', 'text-emerald-200',
    'border-rose-400/30', 'bg-rose-400/10', 'text-rose-200',
    'border-emerald-400/25', 'border-sky-400/25',
    'bg-rose-500', 'bg-amber-400', 'text-rose-300', 'text-amber-300', 'text-sky-300'
) | ForEach-Object { $classes.Add($_) }

$custom = @('panel', 'drop', 'step', 'dot', 'label', 'core', 'halo',
    'success-panel', 'success-glow', 'success-circle', 'success-tick', "'", '+')
$missing = @()
foreach ($c in ($classes | Sort-Object -Unique)) {
    if ($custom -contains $c) { continue }
    $pattern = '(?<![\w-])\.' + [regex]::Escape($c) + '(?=[,{:\s>])'
    if (-not [regex]::IsMatch($styleCss, $pattern)) { $missing += $c }
}
"classes checked     : " + (($classes | Sort-Object -Unique).Count)
"classes unstyleable : " + $missing.Count
if ($missing.Count -gt 0) { $missing | ForEach-Object { "  - $_" }; $problems.Add('missing utility rules') }

# 4. Expected hooks still present
foreach ($needle in @('id="duckForm"', 'id="duckVol"', 'id="duckBtn"', 'id="demoOriginal"', 'id="demoDucked"',
        'id="player"', 'id="planBadge"', 'id="upgradeBtn"', 'id="trialBar"', 'id="trialText"',
        'id="feedBadge"', 'id="lengthModal"', 'id="successModal"', 'data-track="dialogue"',
        'data-track="music"', 'data-lucide="mic"', 'data-lucide="music"', 'data-lucide="clock"',
        '/api/billing/status', '/api/billing/checkout', '/api/billing/portal', '/api/billing/verify')) {
    if ($html -notlike "*$needle*") { $problems.Add("missing hook: $needle") }
}

"---"
if ($problems.Count -eq 0) { "VALIDATION OK" } else { "VALIDATION FAILED"; $problems | ForEach-Object { "  * $_" } }
