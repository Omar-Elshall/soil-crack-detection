# laptop_autoconnect.ps1 — one-shot, self-elevating PowerShell watcher.
#
# Run from a PowerShell prompt (or double-click via the wrapper .bat).
# UAC will prompt ONCE; after that the script needs no further input.
#
# What it does:
#   1. Probes the Jetson on shared WiFi (via wsl.exe ssh)
#   2. If unreachable: reads laptop's current WiFi SSID + password (admin netsh),
#      switches laptop to the Jetson's hotspot, ssh's to 10.42.0.1, pushes the
#      creds, switches laptop back, updates ~/.ssh/config inside WSL with the
#      Jetson's new IP.
#   3. Once reachable: opens the default browser to the live UI.
#   4. Loops every 3 s. 90 s cooldown between bootstrap attempts.
#
# Why PowerShell instead of WSL bash:
#   - Native admin via Start-Process -Verb RunAs (UAC, one prompt)
#   - netsh runs in this process, no shelling out
#   - SSH still goes through wsl.exe so it reuses your existing ~/.ssh/config

# ── Self-elevate ────────────────────────────────────────────────────────────
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Re-launching elevated (UAC prompt)..." -ForegroundColor Yellow
    Start-Process powershell.exe "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
    exit
}

$ErrorActionPreference = "Continue"

# ── Constants ───────────────────────────────────────────────────────────────
$JETSON_HOTSPOT_SSID = "soil-crack-demo"
$JETSON_HOTSPOT_PASS = "cracksoil2026"
$JETSON_HOTSPOT_IP   = "10.42.0.1"
$SSH_USER            = "sdp-w-nano"
$URL_DEFAULT         = "http://soilcrack.local:5173"
$BOOTSTRAP_COOLDOWN  = 90  # seconds

# ── Helpers ─────────────────────────────────────────────────────────────────
function Wsl-Ssh {
    param([string]$Target, [string]$Cmd)
    # Run ssh from inside WSL so it uses ~/.ssh/config + jetson_nano key
    $r = & wsl.exe -e bash -c "ssh -o ConnectTimeout=2 -o BatchMode=yes $Target '$Cmd' 2>/dev/null; echo `$?"
    return ($r[-1] -eq "0")
}

function Get-CurrentSsid {
    $info = netsh wlan show interfaces 2>$null
    if ($info -match '^\s*SSID\s*:\s*(.+?)\s*$') { return $matches[1] }
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
        # Force-replace (in case it was added in auto mode previously)
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
            if (Wsl-Ssh "jetson" "echo up") { return $t }
        } else {
            if (Wsl-Ssh "$SSH_USER@$t" "echo up") { return $t }
        }
    }
    return $null
}

function Bootstrap-ViaHotspot {
    $ssid = Get-CurrentSsid
    if (-not $ssid) {
        Write-Host "    No WiFi connected — skip bootstrap." -ForegroundColor Yellow
        return $false
    }
    Write-Host "    Laptop on: " -NoNewline; Write-Host $ssid -ForegroundColor Green

    $pass = Get-WifiPassword $ssid
    if (-not $pass) {
        Write-Host "    Could not read password for '$ssid' — abort." -ForegroundColor Red
        return $false
    }

    Ensure-HotspotProfile
    Write-Host "    Switching laptop → $JETSON_HOTSPOT_SSID"
    netsh wlan connect name="$JETSON_HOTSPOT_SSID" >$null 2>&1

    Write-Host -NoNewline "    Waiting for Jetson on hotspot"
    $reached = $false
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 2
        if (Wsl-Ssh "$SSH_USER@$JETSON_HOTSPOT_IP" "echo up") { $reached = $true; break }
        Write-Host -NoNewline "."
    }
    Write-Host ""
    if (-not $reached) {
        Write-Host "    Could not reach hotspot — restoring laptop WiFi." -ForegroundColor Red
        netsh wlan connect name="$ssid" >$null 2>&1
        return $false
    }

    Write-Host "    Pushing creds to Jetson..."
    $pushCmd = "sudo nmcli connection modify Hotspot connection.autoconnect-priority 1 2>/dev/null; nohup sudo bash -c 'sleep 4 && nmcli device wifi connect `"$ssid`" password `"$pass`"' > /tmp/wifi-share.log 2>&1 & disown; exit 0"
    & wsl.exe -e bash -c "ssh $SSH_USER@$JETSON_HOTSPOT_IP `"$pushCmd`" 2>/dev/null" | Out-Null

    Write-Host "    Switching laptop back → $ssid"
    Start-Sleep -Seconds 4
    netsh wlan connect name="$ssid" >$null 2>&1

    # Find Jetson's new IP via Bonjour/mDNS, update WSL ~/.ssh/config
    Write-Host -NoNewline "    Resolving Jetson on $ssid"
    $newIp = $null
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 3
        try {
            $ans = Resolve-DnsName -Name "soilcrack.local" -ErrorAction SilentlyContinue | Where-Object {$_.Type -eq "A"} | Select-Object -First 1
            if ($ans) {
                $candIp = $ans.IPAddress
                if (Wsl-Ssh "$SSH_USER@$candIp" "echo up") { $newIp = $candIp; break }
            }
        } catch {}
        Write-Host -NoNewline "."
    }
    Write-Host ""
    if (-not $newIp) {
        Write-Host "    Bootstrap incomplete — Jetson not yet visible on $ssid" -ForegroundColor Yellow
        return $false
    }

    Write-Host "    Jetson at " -NoNewline; Write-Host $newIp -ForegroundColor Green
    # Update ~/.ssh/config inside WSL
    $sedCmd = "if grep -q '^Host jetson$' ~/.ssh/config 2>/dev/null; then sed -i '/^Host jetson$/,/^Host /{s/^[[:space:]]*HostName .*/\tHostName $newIp/}' ~/.ssh/config; else mkdir -p ~/.ssh && chmod 700 ~/.ssh && printf '\nHost jetson\n\tHostName $newIp\n\tUser $SSH_USER\n\tIdentityFile ~/.ssh/jetson_nano\n' >> ~/.ssh/config && chmod 600 ~/.ssh/config; fi"
    & wsl.exe -e bash -c $sedCmd | Out-Null
    Write-Host "    Updated WSL ~/.ssh/config — 'ssh jetson' tracks $newIp" -ForegroundColor Green
    return $true
}

# ── Main loop ───────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "==> Auto-onboard watcher (PowerShell, elevated)" -ForegroundColor Green
Write-Host "    Detect → bootstrap if needed → open browser. Ctrl+C to stop."
Write-Host ""

$lastBootstrap = 0
$wasUp = $false
$url = ""
while ($true) {
    $detected = Probe-Jetson
    if ($detected) {
        if (-not $wasUp) {
            Write-Host ("[" + (Get-Date -Format "HH:mm:ss") + "] ") -NoNewline
            Write-Host "Jetson detected" -ForegroundColor Green -NoNewline
            Write-Host " via $detected"
            if ($detected -eq "192.168.55.1") { $url = "http://192.168.55.1:5173" } else { $url = $URL_DEFAULT }
            Start-Process $url
            Write-Host "    Browser opened → " -NoNewline; Write-Host $url -ForegroundColor Green
            $wasUp = $true
        }
    } else {
        if ($wasUp) {
            Write-Host ("[" + (Get-Date -Format "HH:mm:ss") + "] ") -NoNewline
            Write-Host "Jetson unreachable" -ForegroundColor Yellow
            $wasUp = $false
        }
        $now = [int][double]::Parse((Get-Date -UFormat %s))
        if (($now - $lastBootstrap) -ge $BOOTSTRAP_COOLDOWN) {
            $lastBootstrap = $now
            Write-Host ("[" + (Get-Date -Format "HH:mm:ss") + "] ") -NoNewline
            Write-Host "Jetson unreachable — bootstrapping via hotspot" -ForegroundColor Yellow
            Bootstrap-ViaHotspot | Out-Null
        }
    }
    Start-Sleep -Seconds 3
}
