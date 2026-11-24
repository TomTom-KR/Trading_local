# tutorial-speak.ps1 - Sprechertext des Video-Tutorials mit der Windows-Stimme (SAPI) in WAV-Dateien umwandeln.
#
# Aufruf (von tutorial.mjs): powershell -NoProfile -File tutorial-speak.ps1 -Json <liste.json> -OutDir <ordner>
#   <liste.json>: [{"id": "intro", "text": "Willkommen ..."}, ...] (UTF-8)
#   Ergebnis: <ordner>\<id>.wav je Eintrag.
param(
  [Parameter(Mandatory = $true)][string]$Json,
  [Parameter(Mandatory = $true)][string]$OutDir,
  [string]$Voice = 'Microsoft Hedda Desktop',
  [int]$Rate = -1
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$items = Get-Content -Raw -Encoding UTF8 -LiteralPath $Json | ConvertFrom-Json
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  $synth.SelectVoice($Voice)
} catch {
  $names = ($synth.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }) -join ', '
  throw "Stimme '$Voice' ist nicht installiert. Verfügbar: $names"
}
$synth.Rate = $Rate
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
foreach ($item in $items) {
  $file = Join-Path $OutDir ($item.id + '.wav')
  $synth.SetOutputToWaveFile($file)
  $synth.Speak([string]$item.text)
  $synth.SetOutputToNull()
  Write-Output ("{0}`t{1}" -f $item.id, $file)
}
$synth.Dispose()
