#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
USER_NAME="${SUDO_USER:-$(id -un)}"
[[ "$USER_NAME" != root ]] || { echo 'Run with sudo from the runtime user account.' >&2; exit 1; }
[[ "$ROOT" != *' '* ]] || { echo 'Install path must not contain spaces.' >&2; exit 1; }
test -f "$ROOT/local/params.toml"
for unit in microduck-control microduck-app; do
  if systemctl is-active --quiet "$unit"; then
    echo "$unit is running. Stop the old stack explicitly before installing." >&2; exit 1
  fi
done
if systemctl is-active --quiet microduck-hd1910; then
  echo 'Release service is active; stop it explicitly before replacing.' >&2; exit 1
fi
printf '%s\n' '[Unit]' 'Description=MicroDuck HD1910 native runtime and APK backend' \
  'After=network.target' '[Service]' "User=$USER_NAME" "WorkingDirectory=$ROOT" \
  "ExecStart=/bin/bash $ROOT/scripts/run_hardware.sh" 'Restart=no' 'KillMode=control-group' \
  'TimeoutStopSec=20' '[Install]' 'WantedBy=multi-user.target' \
  | install -m644 /dev/stdin /etc/systemd/system/microduck-hd1910.service
systemctl daemon-reload
systemctl enable microduck-hd1910
echo 'Installed/enabled, NOT started. Start explicitly after checking calibration and device permissions.'
