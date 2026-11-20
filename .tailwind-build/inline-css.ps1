$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$indexPath = Join-Path $root 'index.html'
$cssPath = Join-Path $PSScriptRoot 'out.css'
$startTok = '/* __TAILWIND_BUILD_START__ */'
$endTok = '/* __TAILWIND_BUILD_END__ */'

# IMPORTANT: index.html is UTF-8 (it contains emoji and typographic dashes).
# Always decode/encode explicitly - Windows PowerShell's default Get-Content
# uses the ANSI codepage and silently mangles multi-byte characters.
$bytes = [System.IO.File]::ReadAllBytes($indexPath)
$hasBom = ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF)
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$reader = New-Object System.IO.StreamReader($indexPath, $utf8NoBom, $true)
$html = $reader.ReadToEnd()
$reader.Dispose()

# Baseline: decoding must be lossless, so the non-ASCII count may not change.
$nonAsciiBefore = ([regex]::Matches($html, '[^\x00-\x7F]')).Count
if ($nonAsciiBefore -eq 0) { throw 'Non-ASCII characters vanished - encoding was mishandled' }

$css = [System.IO.File]::ReadAllText($cssPath, $utf8NoBom).Trim()

if ($css.Contains('</style')) { throw 'Compiled CSS must not contain a style end tag' }
if ($css.Contains($endTok)) { throw 'Compiled CSS must not contain the build end marker' }

$si = $html.IndexOf($startTok)
$ei = $html.IndexOf($endTok)
if ($si -lt 0) { throw "Build start marker not found in index.html" }
if ($ei -lt $si) { throw "Build end marker must come after the start marker" }

$nl = if ($html.Contains("`r`n")) { "`r`n" } else { "`n" }
$before = $html.Substring(0, $si + $startTok.Length)
$after = $html.Substring($ei)
$html = $before + $nl + $css + $nl + $after

[System.IO.File]::WriteAllText($indexPath, $html, (New-Object System.Text.UTF8Encoding($hasBom)))

# Re-read and confirm nothing was mangled
$check = [System.IO.File]::ReadAllText($indexPath, $utf8NoBom)
$nonAsciiAfter = ([regex]::Matches($check, '[^\x00-\x7F]')).Count
if ($nonAsciiAfter -ne $nonAsciiBefore) { throw "Encoding drift: $nonAsciiBefore -> $nonAsciiAfter non-ASCII chars" }

"inlined {0:N0} characters of CSS into index.html (non-ASCII chars intact: {1})" -f $css.Length, $nonAsciiAfter
