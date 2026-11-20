param(
    # ===================== EDIT THESE =====================
    # "ship"              = current best: scripted bot on bot/decks/tournament.json
    # "rollback_scripted" = previous bot: scripted v0 on bot/decks/default.json
    [string]$Mode = "ship",
    [string]$Url  = "wss://ARENA-HOST-HERE",
    [string]$Name = "TEAM-NAME-HERE",
    # 0 = restart forever; N = stop after N restarts (used by tests)
    [int]$MaxRestarts = 0
)
# ======================================================
# Live bot launcher with auto-restart (handoff C17). Windows PowerShell 5.1 compatible.
#
#   powershell -ExecutionPolicy Bypass -File .\run_bot.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_bot.ps1 -Mode rollback_scripted
#
# Runs bot.client in this window; whenever it exits (crash, killed, network) it logs the exit to
# runs\run_bot.log and starts it again after $RestartDelay seconds. Ctrl+C stops everything.
# To roll back: set $Mode above to "rollback_scripted" and restart this script.

$RestartDelay = 5

Set-Location -LiteralPath $PSScriptRoot

# --- mode -> client arguments ---
switch ($Mode) {
    "ship" {
        $Policy = "scripted"
        $Deck = "bot/decks/tournament.json"
    }
    "rollback_scripted" {
        $Policy = "class:bot.scripted_v0.ScriptedPolicyV0"
        $Deck = "bot/decks/default.json"
    }
    # Future learned mode (no model yet), e.g. rolling back to the previous best checkpoint:
    # "rollback_learned" { $Policy = "learned:models/prev_best.pt"; $Deck = "bot/decks/tournament.json" }
    default {
        Write-Host "Unknown -Mode '$Mode'. Use ship or rollback_scripted." -ForegroundColor Red
        exit 2
    }
}

if ($Url -like "*ARENA-HOST-HERE*" -or $Name -like "*TEAM-NAME-HERE*") {
    Write-Host "Set `$Url and `$Name at the top of run_bot.ps1 (or pass -Url and -Name) first." -ForegroundColor Red
    Write-Host "  Url:  $Url"
    Write-Host "  Name: $Name"
    exit 1
}

$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Py)) {
    $Alt = Join-Path (Join-Path (Join-Path $PSScriptRoot ".venv") "bin") "python"  # macOS / Linux venv
    if (Test-Path -LiteralPath $Alt) {
        $Py = $Alt
    } else {
        Write-Host "No virtual environment found at $Py" -ForegroundColor Red
        Write-Host "Run the installer first: powershell -ExecutionPolicy Bypass -File .\install.ps1"
        exit 1
    }
}

$RunsDir = Join-Path $PSScriptRoot "runs"
if (-not (Test-Path -LiteralPath $RunsDir)) { New-Item -ItemType Directory -Path $RunsDir | Out-Null }
$LogFile = Join-Path $RunsDir "run_bot.log"

function Write-Log([string]$Msg, [string]$Color = "Yellow") {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Msg
    Write-Host $line -ForegroundColor $Color
    [System.IO.File]::AppendAllText($LogFile, $line + [Environment]::NewLine)
}

$ClientArgs = @(
    "-m", "bot.client",
    "--url", $Url,
    "--name", $Name,
    "--policy", $Policy,
    "--deck", $Deck,
    "--log", "runs/client.log",
    "--summary-json", "runs/live_summary.json"
)

Write-Host ""
Write-Host "Before leaving the bot running:" -ForegroundColor Cyan
Write-Host "  - plug the laptop in"
Write-Host "  - no sleep while plugged in:  powercfg /change standby-timeout-ac 0"
Write-Host "  - pause OneDrive syncing (it can lock files under runs\)"
Write-Host "  Ctrl+C in this window stops the bot and the restart loop."
Write-Host ""

$Restarts = 0
try {
    while ($true) {
        Write-Log ("starting client: mode {0} | policy {1} | deck {2} | name {3} | url {4}" -f $Mode, $Policy, $Deck, $Name, $Url) "Green"
        & $Py @ClientArgs
        $Code = $LASTEXITCODE
        if ($MaxRestarts -gt 0 -and $Restarts -ge $MaxRestarts) {
            Write-Log ("client exited with code {0}; reached -MaxRestarts {1}, stopping" -f $Code, $MaxRestarts)
            break
        }
        Write-Log ("client exited with code {0}, restarting in {1} s" -f $Code, $RestartDelay)
        Start-Sleep -Seconds $RestartDelay
        $Restarts++
    }
} finally {
    Write-Log "run_bot.ps1 stopped" "Cyan"
}
