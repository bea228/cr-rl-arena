param(
    [int]$Matches = 200,   # 1000 for the full C16 run
    [int]$Port = 8799
)
# Soak test (handoff C16). Windows PowerShell 5.1 compatible.
#
#   powershell -ExecutionPolicy Bypass -File scripts\soak.ps1 -Matches 200
#
# Local server on 127.0.0.1:<Port> with a fresh data folder under runs\soak\, and two clients via
# python -m bot.client: soak-ship (scripted, tournament deck) vs soak-rollback (scripted v0, default
# deck), each with --max-matches <Matches>. When soak-ship is about half done the server is killed and
# started again on the same data folder; the clients must reconnect on their own.
# PASS: both clients finish, 0 missed, 0 malformed, other ignored moves < 1% of decisions.

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Repo

$Py = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Py)) {
    $Alt = Join-Path (Join-Path (Join-Path $Repo ".venv") "bin") "python"  # macOS / Linux venv
    if (Test-Path -LiteralPath $Alt) { $Py = $Alt } else { Write-Host "No .venv python found; run install.ps1 first." -ForegroundColor Red; exit 1 }
}
$IsWin = ($env:OS -eq "Windows_NT")

$SoakDir = Join-Path "runs" "soak"
New-Item -ItemType Directory -Force -Path $SoakDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$DataDir = Join-Path $SoakDir "data-$Stamp"
$SoakLog = Join-Path $SoakDir "soak-$Stamp.log"
$Url = "ws://127.0.0.1:$Port"
$Timeout = $Matches * 30
$Started = @()   # every process this script started (only these get killed)

function Write-Log([string]$Msg, [string]$Color = "Gray") {
    $line = "{0} {1}" -f (Get-Date -Format "HH:mm:ss"), $Msg
    Write-Host $line -ForegroundColor $Color
    [System.IO.File]::AppendAllText((Join-Path $Repo $SoakLog), $line + [Environment]::NewLine)
}

function Stop-Tree($Proc) {
    # On Windows the venv python.exe is a launcher with a child interpreter: kill the whole tree.
    if ($null -eq $Proc -or $Proc.HasExited) { return }
    if ($IsWin) {
        & taskkill.exe /PID $Proc.Id /T /F 2>&1 | Out-Null
    } else {
        Stop-Process -Id $Proc.Id -Force -ErrorAction SilentlyContinue
    }
    try { $Proc.WaitForExit(10000) | Out-Null } catch { }
}

function Test-Port([int]$P) {
    $c = New-Object System.Net.Sockets.TcpClient
    try { $c.Connect("127.0.0.1", $P); return $true } catch { return $false } finally { $c.Close() }
}

function Start-Server([string]$Tag) {
    if (Test-Port $Port) { throw "port $Port is already in use; pick another with -Port" }
    $old = @{ B = $env:CR_ARENA_BIND; D = $env:CR_ARENA_DATA }
    $env:CR_ARENA_BIND = "127.0.0.1:$Port"
    $env:CR_ARENA_DATA = $DataDir
    try {
        $p = Start-Process -FilePath $Py -ArgumentList @("-m", "cr_server.server") -WorkingDirectory $Repo `
            -NoNewWindow -PassThru `
            -RedirectStandardOutput (Join-Path $SoakDir "server-$Stamp-$Tag.out") `
            -RedirectStandardError (Join-Path $SoakDir "server-$Stamp-$Tag.err")
    } finally {
        $env:CR_ARENA_BIND = $old.B
        $env:CR_ARENA_DATA = $old.D
    }
    $deadline = (Get-Date).AddSeconds(30)
    while (-not (Test-Port $Port)) {
        if ($p.HasExited) { throw "server exited during startup (see $SoakDir\server-$Stamp-$Tag.err)" }
        if ((Get-Date) -gt $deadline) { throw "server did not open port $Port within 30 s" }
        Start-Sleep -Milliseconds 300
    }
    return $p
}

function Start-Client([string]$Name, [string]$Policy, [string]$Deck) {
    $summary = Join-Path $SoakDir "$Name.json"
    $log = Join-Path $SoakDir "$Name.log"
    foreach ($f in @($summary, $log)) { if (Test-Path -LiteralPath $f) { Remove-Item -LiteralPath $f -Force } }
    $args_ = @("-m", "bot.client", "--url", $Url, "--name", $Name, "--policy", $Policy, "--deck", $Deck,
               "--max-matches", "$Matches", "--summary-json", $summary, "--log", $log)
    return Start-Process -FilePath $Py -ArgumentList $args_ -WorkingDirectory $Repo -NoNewWindow -PassThru `
        -RedirectStandardOutput (Join-Path $SoakDir "$Name-$Stamp.out") `
        -RedirectStandardError (Join-Path $SoakDir "$Name-$Stamp.err")
}

function Read-Summary([string]$Name) {
    $f = Join-Path $SoakDir "$Name.json"
    if (-not (Test-Path -LiteralPath $f)) { return $null }
    try { return (Get-Content -LiteralPath $f -Raw | ConvertFrom-Json) } catch { return $null }
}

function Get-Count($Summary, [string]$Key) {
    if ($null -eq $Summary) { return 0 }
    $v = $Summary.last_action.$Key
    if ($null -eq $v) { return 0 }
    return [int]$v
}

$Clients = @(
    @{ Name = "soak-ship";     Policy = "scripted";                              Deck = "bot/decks/tournament.json" },
    @{ Name = "soak-rollback"; Policy = "class:bot.scripted_v0.ScriptedPolicyV0"; Deck = "bot/decks/default.json" }
)

$T0 = Get-Date
$Restarted = $false
$RestartAt = [int][math]::Floor($Matches / 2)
$TimedOut = $false
try {
    Write-Log ("soak: {0} matches per client, server {1}, data {2}, timeout {3} s" -f $Matches, $Url, $DataDir, $Timeout) "Cyan"
    $Server = Start-Server "a"
    $Started += $Server
    Write-Log ("server started (pid {0})" -f $Server.Id)
    $Procs = @{}
    foreach ($c in $Clients) {
        $p = Start-Client $c.Name $c.Policy $c.Deck
        $Procs[$c.Name] = $p
        $Started += $p
        Write-Log ("{0} started (pid {1}): policy {2}, deck {3}" -f $c.Name, $p.Id, $c.Policy, $c.Deck)
    }

    while ($true) {
        Start-Sleep -Seconds 10
        $elapsed = [int]((Get-Date) - $T0).TotalSeconds
        $parts = @()
        foreach ($c in $Clients) {
            $s = Read-Summary $c.Name
            if ($null -eq $s) { $parts += ("{0}: 0" -f $c.Name); continue }
            $parts += ("{0}: {1}/{2} W{3} L{4} D{5} missed {6} malformed {7}" -f $c.Name, $s.matches, $Matches,
                $s.wins, $s.losses, $s.draws, (Get-Count $s "missed"), (Get-Count $s "malformed"))
        }
        Write-Log ("[{0,5} s] {1}" -f $elapsed, ($parts -join " | "))

        $ship = Read-Summary "soak-ship"
        if (-not $Restarted -and $null -ne $ship -and $ship.matches -ge $RestartAt) {
            Write-Log ("soak-ship at {0} matches: killing the server (pid {1}) and starting it again" -f $ship.matches, $Server.Id) "Yellow"
            Stop-Tree $Server
            $Server = Start-Server "b"
            $Started += $Server
            $Restarted = $true
            Write-Log ("server restarted (pid {0}), same data folder; clients must reconnect on their own" -f $Server.Id) "Yellow"
        }

        $running = @($Procs.Values | Where-Object { -not $_.HasExited })
        if ($running.Count -eq 0) { Write-Log "both clients exited"; break }
        if ($elapsed -ge $Timeout) { Write-Log ("timeout after {0} s" -f $elapsed) "Red"; $TimedOut = $true; break }
    }
} finally {
    foreach ($p in $Started) { Stop-Tree $p }
}

# --- report ---
$Wall = ((Get-Date) - $T0).TotalSeconds
$AllPass = $Restarted -and -not $TimedOut
Write-Host ""
Write-Log ("=== soak report: {0} matches per client, wall time {1:N0} s ({2:N2} s per match), server restart {3} ===" -f `
    $Matches, $Wall, ($Wall / [math]::Max(1, $Matches)), $(if ($Restarted) { "done" } else { "NOT done" })) "Cyan"
foreach ($c in $Clients) {
    $s = Read-Summary $c.Name
    if ($null -eq $s) { Write-Log ("{0}: no summary written -> FAIL" -f $c.Name) "Red"; $AllPass = $false; continue }
    $missed = Get-Count $s "missed"
    $malformed = Get-Count $s "malformed"
    $other = [int]$s.ignored - $missed - $malformed
    $share = 0.0
    if ($s.decisions -gt 0) { $share = $other / [double]$s.decisions }
    $counts = ($s.last_action.PSObject.Properties | ForEach-Object { "{0}={1}" -f $_.Name, $_.Value }) -join ", "
    $ok = ($s.matches -ge $Matches) -and ($missed -eq 0) -and ($malformed -eq 0) -and ($share -lt 0.01)
    if (-not $ok) { $AllPass = $false }
    Write-Log ("{0}: matches {1}  W/L/D {2}/{3}/{4}  aborted {5}  rating {6}" -f $c.Name, $s.matches, $s.wins, $s.losses, $s.draws, $s.aborted, $s.rating)
    Write-Log ("    last_action {{{0}}}" -f $counts)
    Write-Log ("    missed {0}  malformed {1}  other ignored {2}/{3} decisions = {4:P2}  slow {5}  policy errors {6}  -> {7}" -f `
        $missed, $malformed, $other, $s.decisions, $share, $s.slow_decisions, $s.policy_errors, $(if ($ok) { "PASS" } else { "FAIL" })) $(if ($ok) { "Green" } else { "Red" })
}
if ($AllPass) {
    Write-Log "C16 soak: PASS (0 missed, 0 malformed, other ignored < 1%, server restart survived)" "Green"
    exit 0
}
Write-Log "C16 soak: FAIL (see above; logs in $SoakDir)" "Red"
exit 1
