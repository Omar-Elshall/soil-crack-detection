# laptop_autoconnect.ps1 - one-shot, self-elevating PowerShell watcher.
#
# Run from a PowerShell prompt (or via install_laptop_autostart.bat).
# UAC will prompt ONCE; after that the script needs no further input.
#
# What it does:
#   1. Probes the Jetson on shared WiFi (via Windows-native ssh.exe)
#   2. If unreachable: reads laptop's current WiFi SSID + password (admin
#      netsh), switches laptop to the Jetson's hotspot, ssh's to 10.42.0.1,
#      pushes the creds, switches laptop back, updates Windows ssh config
#      with the Jetson's new IP.
#   3. Once reachable: opens the default browser to the live UI.
#   4. Loops every 3 s. 90 s cooldown between bootstrap attempts.

# --- Self-elevate ----------------------------------------------------------
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Re-launching elevated (UAC prompt)..." -ForegroundColor Yellow
    Start-Process powershell.exe "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
    exit
}

$ErrorActionPreference = "Continue"

# --- Constants -------------------------------------------------------------
$JETSON_HOTSPOT_SSID = "soil-crack-demo"
$JETSON_HOTSPOT_PASS = "cracksoil2026"
$JETSON_HOTSPOT_IP   = "10.42.0.1"
$SSH_USER            = "sdp-w-nano"
$URL_DEFAULT         = "http://soilcrack.local:5173"
$BOOTSTRAP_COOLDOWN_OK   = 90    # cooldown after a successful bootstrap
$BOOTSTRAP_COOLDOWN_FAIL = 20    # cooldown after a failed attempt (Jetson likely still booting)

$SSH_EXE = "C:\Windows\System32\OpenSSH\ssh.exe"
if (-not (Test-Path $SSH_EXE)) { $SSH_EXE = "ssh.exe" }
$SSH_KEY = Join-Path $env:USERPROFILE ".ssh\jetson_nano"

# --- Helpers ---------------------------------------------------------------
function Win-Ssh {
    param([string]$Target, [string]$Cmd)
    # ConnectTimeout caps TCP connect; ServerAlive* kills hung sessions in
    # ~3s. Combined cap per probe ~3s even on dead routes.
    $args = @(
        "-o", "ConnectTimeout=2",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=2",
        "-o", "ServerAliveCountMax=1"
    )
    if ($Target -eq "jetson") {
        & $SSH_EXE @args $Target $Cmd *>$null
    } else {
        $args += @("-i", $SSH_KEY, "-o", "UserKnownHostsFile=NUL")
        & $SSH_EXE @args $Target $Cmd *>$null
    }
    return ($LASTEXITCODE -eq 0)
}

function Get-CurrentSsid {
    $info = netsh wlan show interfaces 2>$null
    if (-not $info) { return $null }
    foreach ($line in $info) {
        # match exactly the SSID line (not BSSID, which also contains "SSID")
        if ($line -match '^\s*SSID\s+:\s+(.+?)\s*$') { return $matches[1] }
    }
    return $null
}

function Get-WifiPassword {
    param([string]$Ssid)
    $info = netsh wlan show profile name="$Ssid" key=clear 2>$null
    foreach ($line in $info) {
        if ($line -match '^\s*Key Content\s*:\s*(.+?)\s*$') { return $matches[1] }
    }
    return $null
}

function Ensure-HotspotProfile {
    $profiles = netsh wlan show profiles 2>$null
    if ($profiles -match [regex]::Escape($JETSON_HOTSPOT_SSID)) {
        netsh wlan delete profile name="$JETSON_HOTSPOT_SSID" >$null 2>&1
    }
    $xml = @"
<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
<name>$JETSON_HOTSPOT_SSID</name>
<SSIDConfig><SSID><name>$JETSON_HOTSPOT_SSID</name></SSID></SSIDConfig>
<connectionType>ESS</connectionType>
<connectionMode>manual</connectionMode>
<MSM><security>
<authEncryption><authentication>WPA2PSK</authentication><encryption>AES</encryption><useOneX>false</useOneX></authEncryption>
<sharedKey><keyType>passPhrase</keyType><protected>false</protected><keyMaterial>$JETSON_HOTSPOT_PASS</keyMaterial></sharedKey>
</security></MSM>
</WLANProfile>
"@
    $tmp = [IO.Path]::GetTempFileName() + ".xml"
    Set-Content -Path $tmp -Value $xml -Encoding ASCII
    netsh wlan add profile filename="$tmp" >$null 2>&1
    Remove-Item $tmp -ErrorAction SilentlyContinue
}

function Probe-Jetson {
    foreach ($t in @("192.168.55.1", "jetson", "soilcrack.local")) {
        if ($t -eq "jetson") {
            if (Win-Ssh "jetson" "echo up") { return $t }
        } else {
            if (Win-Ssh "$SSH_USER@$t" "echo up") { return $t }
        }
    }
    return $null
}

function Bootstrap-ViaHotspot {
    $ssid = Get-CurrentSsid
    if (-not $ssid) {
        Write-Host "    No WiFi connected - skip bootstrap." -ForegroundColor Yellow
        return $false
    }
    Write-Host "    Laptop on: " -NoNewline; Write-Host $ssid -ForegroundColor Green

    $pass = Get-WifiPassword $ssid
    if (-not $pass) {
        Write-Host "    Could not read password for '$ssid' - abort." -ForegroundColor Red
        return $false
    }

    Ensure-HotspotProfile
    Write-Host "    Switching laptop -> $JETSON_HOTSPOT_SSID"
    netsh wlan connect name="$JETSON_HOTSPOT_SSID" >$null 2>&1

    Write-Host -NoNewline "    Waiting for Jetson on hotspot"
    $reached = $false
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 2
        if (Win-Ssh "$SSH_USER@$JETSON_HOTSPOT_IP" "echo up") { $reached = $true; break }
        Write-Host -NoNewline "."
    }
    Write-Host ""
    if (-not $reached) {
        Write-Host "    Could not reach hotspot - restoring laptop WiFi." -ForegroundColor Red
        netsh wlan connect name="$ssid" >$null 2>&1
        return $false
    }

    Write-Host "    Pushing creds to Jetson..."
    # Bash command we ship to the Jetson:
    #   1. lower hotspot autoconnect priority so it stays out of the way
    #   2. backgrounded: drop hotspot, wait for radio to switch out of AP
    #      mode, rescan (REQUIRED — radio in AP mode shows no scan results),
    #      then nmcli connect to the new SSID. Backgrounded with nohup so the
    #      SSH session can return cleanly before the WiFi link drops.
    $bashTemplate = @'
sudo nmcli connection modify Hotspot connection.autoconnect-priority 1 2>/dev/null; nohup sudo bash -c 'sleep 2 && nmcli connection down Hotspot && sleep 3 && nmcli device wifi rescan && sleep 8 && nmcli device wifi connect "@@SSID@@" password "@@PASS@@"' > /tmp/wifi-share.log 2>&1 & disown; exit 0
'@
    $pushCmd = $bashTemplate.Replace('@@SSID@@', $ssid).Replace('@@PASS@@', $pass)
    & $SSH_EXE -i $SSH_KEY -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL "$SSH_USER@$JETSON_HOTSPOT_IP" $pushCmd *>$null

    Write-Host "    Switching laptop back -> $ssid"
    Start-Sleep -Seconds 4
    netsh wlan connect name="$ssid" >$null 2>&1

    # Find Jetson's new IP via Bonjour, update Windows ssh config.
    Write-Host -NoNewline "    Resolving Jetson on $ssid"
    $newIp = $null
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 3
        try {
            $ans = Resolve-DnsName -Name "soilcrack.local" -ErrorAction SilentlyContinue | Where-Object {$_.Type -eq "A"} | Select-Object -First 1
            if ($ans) {
                $candIp = $ans.IPAddress
                if (Win-Ssh "$SSH_USER@$candIp" "echo up") { $newIp = $candIp; break }
            }
        } catch {}
        Write-Host -NoNewline "."
    }
    Write-Host ""
    if (-not $newIp) {
        Write-Host "    Bootstrap incomplete - Jetson not yet visible on $ssid" -ForegroundColor Yellow
        return $false
    }

    Write-Host "    Jetson at " -NoNewline; Write-Host $newIp -ForegroundColor Green

    # Update Windows-side ssh config so 'ssh jetson' tracks the new IP.
    $sshDir = Join-Path $env:USERPROFILE ".ssh"
    if (-not (Test-Path $sshDir)) { New-Item -ItemType Directory -Path $sshDir | Out-Null }
    $configPath = Join-Path $sshDir "config"
    $keyPath = Join-Path $sshDir "jetson_nano"

    $newBlock = "Host jetson`r`n`tHostName $newIp`r`n`tUser $SSH_USER`r`n`tIdentityFile $keyPath`r`n"

    if (Test-Path $configPath) {
        $cfgContent = Get-Content $configPath -Raw
        # Remove any existing Host jetson block, then append the new one.
        $cfgContent = [regex]::Replace($cfgContent, '(?ms)^Host jetson\s*\r?\n(?:[ \t]+\S.*\r?\n?)*', '')
        $cfgContent = $cfgContent.TrimEnd() + "`r`n`r`n" + $newBlock
        Set-Content -Path $configPath -Value $cfgContent -NoNewline
    } else {
        Set-Content -Path $configPath -Value $newBlock
    }
    Write-Host "    Updated $configPath - 'ssh jetson' tracks $newIp" -ForegroundColor Green
    return $true
}

# --- Relay subprocess management -------------------------------------------
# Spawn the laptop_mavlink_relay.py process when Jetson first becomes reachable
# (or in the background even before, since the radio is reachable independently).
# Reaped on script exit via the trap below.
$RELAY_SCRIPT = Join-Path $PSScriptRoot "laptop_mavlink_relay.py"
$RELAY_VENV   = Join-Path $env:USERPROFILE "soilcrack-relay-venv\Scripts\python.exe"
$relayProc    = $null

function Ensure-Relay {
    if ($relayProc -and -not $relayProc.HasExited) { return }
    if (-not (Test-Path $RELAY_SCRIPT)) { return }
    $py = if (Test-Path $RELAY_VENV) { $RELAY_VENV } else { "python.exe" }
    Write-Host ("[" + (Get-Date -Format "HH:mm:ss") + "] ") -NoNewline
    Write-Host "starting MAVLink relay (radio path)..." -ForegroundColor Cyan
    $script:relayProc = Start-Process -FilePath $py `
        -ArgumentList @("-u", $RELAY_SCRIPT) `
        -WindowStyle Hidden `
        -RedirectStandardOutput "$env:TEMP\soilcrack-relay.log" `
        -RedirectStandardError  "$env:TEMP\soilcrack-relay.err" `
        -PassThru
}

# Reap on Ctrl+C / window close
$null = Register-EngineEvent PowerShell.Exiting -Action {
    if ($script:relayProc -and -not $script:relayProc.HasExited) {
        try { $script:relayProc.Kill() } catch {}
    }
}

# --- Main loop -------------------------------------------------------------
Write-Host ""
Write-Host "==> Auto-onboard watcher (PowerShell, elevated)" -ForegroundColor Green
Write-Host "    Detect -> bootstrap if needed -> spawn relay -> open browser."
Write-Host "    Ctrl+C to stop."
Write-Host ""

$lastBootstrap = 0
$lastBootstrapOk = $false
$wasUp = $false
$lastHeartbeat = 0
$url = ""
while ($true) {
    $detected = Probe-Jetson
    $now = [int][double]::Parse((Get-Date -UFormat %s))

    if ($detected) {
        if (-not $wasUp) {
            Write-Host ""
            Write-Host "==> JETSON BACK" -ForegroundColor Green -BackgroundColor Black
            Write-Host ("    [" + (Get-Date -Format "HH:mm:ss") + "] reachable via $detected")
            if ($detected -eq "192.168.55.1") { $url = "http://192.168.55.1:5173" } else { $url = $URL_DEFAULT }
            Ensure-Relay
            Start-Process $url
            Write-Host "    browser -> " -NoNewline; Write-Host $url -ForegroundColor Green
            $wasUp = $true
        }
    } else {
        if ($wasUp) {
            Write-Host ""
            Write-Host "==> JETSON LOST" -ForegroundColor Red -BackgroundColor Black
            Write-Host ("    [" + (Get-Date -Format "HH:mm:ss") + "] no probe target reachable")
            $wasUp = $false
        }
        $cooldown = if ($lastBootstrapOk) { $BOOTSTRAP_COOLDOWN_OK } else { $BOOTSTRAP_COOLDOWN_FAIL }
        $sinceLast = $now - $lastBootstrap
        if ($sinceLast -ge $cooldown) {
            $lastBootstrap = $now
            Write-Host ("[" + (Get-Date -Format "HH:mm:ss") + "] ") -NoNewline
            Write-Host "bootstrapping via hotspot..." -ForegroundColor Yellow
            $lastBootstrapOk = (Bootstrap-ViaHotspot) -eq $true
            $lastHeartbeat = $now
        } elseif (($now - $lastHeartbeat) -ge 6) {
            $remain = $cooldown - $sinceLast
            Write-Host ("[" + (Get-Date -Format "HH:mm:ss") + "] ") -NoNewline
            Write-Host "still polling... (retry bootstrap in ${remain}s)" -ForegroundColor DarkGray
            $lastHeartbeat = $now
        }
    }
    Start-Sleep -Seconds 3
}
