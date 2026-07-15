#!/usr/bin/env bash
# ============================================================================
# ops.sh — Whisper ASR Ops Script
# Supports: Linux (systemd / faster-whisper) + macOS (launchd / mlx-whisper)
# ============================================================================
set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
SERVICE_NAME="whisper-asr-server"
SERVICE_PORT="${WHISPER_PORT:-9080}"
MODEL_SIZE="${WHISPER_MODEL:-medium}"
MODEL_COMPUTE="${WHISPER_COMPUTE:-int8}"
DRY_RUN=false
USER_MODE=false

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --user)    USER_MODE=true ;;
    esac
done

# Set systemd paths per mode
if $USER_MODE; then
    SYSTEMD_UNIT="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/${SERVICE_NAME}.service"
    SYSTEMCTL="systemctl --user"
else
    SYSTEMD_UNIT="/etc/systemd/system/${SERVICE_NAME}.service"
    SYSTEMCTL="systemctl"
fi

# Dry-run helper
run_cmd() { if $DRY_RUN; then echo -e "  ${YELLOW}[DRY-RUN]${NC} $*"; else eval "$@"; fi; }

# macOS launchd
PLIST_NAME="com.whisper.server.plist"
PLIST_PATH="$HOME/Library/LaunchAgents/$PLIST_NAME"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_SCRIPT="$SCRIPT_DIR/whisper_server.py"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERR]${NC}   $*"; }

detect_os() {
    case "$(uname -s)" in
        Linux)  OS="linux" ;;
        Darwin) OS="macos" ;;
        *)      err "Unsupported OS: $(uname -s)"; exit 1 ;;
    esac
}

check_python() {
    if ! command -v python3 &>/dev/null; then
        err "python3 not found. Install Python 3.10+ first"; exit 1
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
# Auto-detect Best Model
# ═══════════════════════════════════════════════════════════════════════════
auto_detect_model() {
    # Skip auto-detect if user explicitly set WHISPER_MODEL
    if [ -n "${WHISPER_MODEL_OVERRIDE:-}" ]; then
        return
    fi

    local ram_gb
    if [ "$OS" = "linux" ]; then
        ram_gb=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
    else
        ram_gb=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1/1024/1024/1024}')
    fi

    if [ "$OS" = "linux" ]; then
        local vram_gb=0
        if command -v nvidia-smi &>/dev/null; then
            vram_gb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{printf "%.0f", $1/1024}')
        fi

        if [ "$vram_gb" -ge 16 ] 2>/dev/null; then
            MODEL_SIZE="large-v3"; MODEL_COMPUTE="float16"
        elif [ "$vram_gb" -ge 8 ] 2>/dev/null; then
            MODEL_SIZE="large-v3-turbo"; MODEL_COMPUTE="float16"
        elif [ "$vram_gb" -ge 4 ] 2>/dev/null; then
            MODEL_SIZE="medium"; MODEL_COMPUTE="float16"
        elif [ "$vram_gb" -gt 0 ] 2>/dev/null; then
            MODEL_SIZE="small"; MODEL_COMPUTE="float16"
        else
            if [ "${ram_gb:-0}" -ge 12 ] 2>/dev/null; then
                MODEL_SIZE="medium"
            elif [ "${ram_gb:-0}" -ge 6 ] 2>/dev/null; then
                MODEL_SIZE="small"
            else
                MODEL_SIZE="base"
            fi
            MODEL_COMPUTE="int8"
        fi

    elif [ "$OS" = "macos" ]; then
        if [ "${ram_gb:-0}" -ge 64 ] 2>/dev/null; then
            MODEL_SIZE="large-v3"
        elif [ "${ram_gb:-0}" -ge 24 ] 2>/dev/null; then
            MODEL_SIZE="large-v3-turbo"
        elif [ "${ram_gb:-0}" -ge 16 ] 2>/dev/null; then
            MODEL_SIZE="medium"
        else
            MODEL_SIZE="small"
        fi
        MODEL_COMPUTE="float16"
    fi
}

# ── Interactive Model Selection ───────────────────────────────────────────────────────
prompt_model() {
    detect_os

    # Auto-detect recommended model
    auto_detect_model
    local recommended="$MODEL_SIZE"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  System: $OS ($(uname -m))"
    if [ "$OS" = "linux" ] && command -v nvidia-smi &>/dev/null; then
        echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'NVIDIA')"
    elif [ "$OS" = "macos" ]; then
        echo "  Chip: $(sysctl -n machdep.cpu.brand_string 2>/dev/null | head -1)"
    fi
    echo "  Memory: $(awk '/MemTotal/ {printf "%.0f GB\n", $2/1024/1024}' /proc/meminfo 2>/dev/null || sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f GB\n", $1/1024/1024/1024}')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Available models
    local models=("tiny" "base" "small" "medium" "large-v3-turbo" "large-v3")
    local labels=(
        "tiny         (39M, fastest, low accuracy)"
        "base         (74M, fast, basic)"
        "small        (244M, fast, good for daily use)"
        "medium       (769M, balanced, high accuracy)"
        "large-v3-turbo (809M, distilled, near-best)"
        "large-v3     (1550M, highest accuracy, needs strong hardware)"
    )

    echo "  Select model (Enter for recommended):"
    echo ""

    for i in "${!models[@]}"; do
        local star=" "
        if [ "${models[$i]}" = "$recommended" ]; then
            star="▶"
            echo -e "  ${GREEN}${star} [$((i+1))] ${labels[$i]}  ← Recommended${NC}"
        else
            echo "     [$((i+1))] ${labels[$i]}"
        fi
    done

    echo ""
    read -r -p "  Enter 1-${#models[@]} or Enter for recommended [$recommended]: " choice

    if [ -n "$choice" ] && [ "$choice" -ge 1 ] 2>/dev/null && [ "$choice" -le "${#models[@]}" ] 2>/dev/null; then
        MODEL_SIZE="${models[$((choice-1))]}"
        if [ "$MODEL_SIZE" != "$recommended" ]; then
            echo ""
            warn "You chose '$MODEL_SIZE'(recommended:  '$recommended'）"
        fi
    else
        MODEL_SIZE="$recommended"
    fi
    echo ""
    info "Final model: $MODEL_SIZE"
}

# ═══════════════════════════════════════════════════════════════════════════
# Model Install
# ═══════════════════════════════════════════════════════════════════════════
model_install() {
    detect_os; check_python
    prompt_model

    if [ "$OS" = "linux" ]; then
        info "Linux → faster-whisper (compute=$MODEL_COMPUTE)"
        if $DRY_RUN; then
            echo "  pip3 install --user faster-whisper fastapi uvicorn python-multipart"
        else
            pip3 install --user faster-whisper fastapi uvicorn python-multipart 2>&1 | tail -3
        fi

        info "downloading model '$MODEL_SIZE'(first run: 3-10 min)..."
        if $DRY_RUN; then
            echo "  python3 -c \"WhisperModel('${MODEL_SIZE}', device='cpu', compute_type='${MODEL_COMPUTE}')\""
        else
            python3 -c "
from faster_whisper import WhisperModel
import time
start = time.time()
m = WhisperModel('${MODEL_SIZE}', device='cpu', compute_type='${MODEL_COMPUTE}')
print(f'Model loaded ✓ ({time.time()-start:.1f}s)')
" 2>&1 | grep -vE 'Warning|HTTP|HEAD|GET|You are'
        fi
        ok "faster-whisper + '$MODEL_SIZE' ready"

    elif [ "$OS" = "macos" ]; then
        if sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -qi "Apple"; then
            info "macOS Apple Silicon → mlx-whisper (GPU)"
            if $DRY_RUN; then
                echo "  pip3 install --user mlx-whisper fastapi uvicorn python-multipart"
            else
                pip3 install --user mlx-whisper fastapi uvicorn python-multipart 2>&1 | tail -3
            fi

            info "downloading model 'mlx-community/whisper-${MODEL_SIZE}'(first run: 2-5 min)..."
            if $DRY_RUN; then
                echo "  python3 -c \"mlx_whisper.transcribe(..., path_or_hf_repo='mlx-community/whisper-${MODEL_SIZE}')\""
            else
                python3 -c "
import mlx_whisper, struct, wave, tempfile, os
with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    with wave.open(f, 'w') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(struct.pack('<h', 0) * 16000)
    f.flush()
    mlx_whisper.transcribe(f.name, path_or_hf_repo='mlx-community/whisper-${MODEL_SIZE}')
    os.unlink(f.name)
print('Model loaded ✓')
" 2>&1 | tail -3
            fi
            ok "mlx-whisper + '$MODEL_SIZE' ready"
        else
            info "macOS Intel → faster-whisper (CPU, compute=$MODEL_COMPUTE)"
            if $DRY_RUN; then
                echo "  pip3 install --user faster-whisper fastapi uvicorn python-multipart"
            else
                pip3 install --user faster-whisper fastapi uvicorn python-multipart 2>&1 | tail -3
            fi

            info "downloading model '$MODEL_SIZE' (first run: 3-10 min)..."
            if $DRY_RUN; then
                echo "  python3 -c \"WhisperModel('${MODEL_SIZE}', device='cpu', compute_type='${MODEL_COMPUTE}')\""
            else
                python3 -c "
from faster_whisper import WhisperModel
import time
start = time.time()
m = WhisperModel('${MODEL_SIZE}', device='cpu', compute_type='${MODEL_COMPUTE}')
print(f'Model loaded ✓ ({time.time()-start:.1f}s)')
" 2>&1 | grep -vE 'Warning|HTTP|HEAD|GET|You are'
            fi
            ok "faster-whisper + '$MODEL_SIZE' ready"
        fi
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
# Service Install — Linux (systemd)
# ═══════════════════════════════════════════════════════════════════════════
service_install_linux() {
    detect_os
    [ "$OS" != "linux" ] && { err "This command is Linux-only"; exit 1; }
    [ ! -f "$SERVER_SCRIPT" ] && { err "not found $SERVER_SCRIPT"; exit 1; }

    local mode_label wanted_by
    if $USER_MODE; then
        mode_label="user-level (systemctl --user)"
        wanted_by="default.target"
    else
        mode_label="system-level (systemctl)"
        wanted_by="multi-user.target"
    fi
    info "Installing systemd unit → $SYSTEMD_UNIT ($mode_label)"

    if $DRY_RUN; then
        echo "  → Will create $SYSTEMD_UNIT"
        echo "  → $SYSTEMCTL daemon-reload && $SYSTEMCTL enable $SERVICE_NAME"
        return
    fi

    if $USER_MODE; then
        mkdir -p "$(dirname "$SYSTEMD_UNIT")"
    fi
    cat > "$SYSTEMD_UNIT" << EOF
[Unit]
Description=Whisper ASR Service (faster-whisper + FastAPI)
After=network.target

[Service]
Type=simple
WorkingDirectory=$SCRIPT_DIR
Environment=PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=WHISPER_MODEL=$MODEL_SIZE
Environment=WHISPER_DEVICE=cpu
Environment=WHISPER_COMPUTE=$MODEL_COMPUTE
Environment=WHISPER_PORT=$SERVICE_PORT
Environment=LLM_API_KEY=
Environment=LLM_BASE_URL=https://api.openai.com/v1
Environment=LLM_MODEL=
Environment=CORRECTION_GAP=2.0
Environment=CORRECTION_CONFIDENCE=-0.5
ExecStart=/usr/bin/python3 $SERVER_SCRIPT
Restart=always
RestartSec=5

[Install]
WantedBy=$wanted_by
EOF
    $SYSTEMCTL daemon-reload
    $SYSTEMCTL enable "$SERVICE_NAME"
    ok "systemd installed & enabled for auto-start ($mode_label)"
}

# ═══════════════════════════════════════════════════════════════════════════
# Service Install — macOS (launchd)
# ═══════════════════════════════════════════════════════════════════════════
service_install_macos() {
    detect_os
    [ "$OS" != "macos" ] && { err "This command is macOS-only"; exit 1; }
    [ ! -f "$SERVER_SCRIPT" ] && { err "not found $SERVER_SCRIPT"; exit 1; }

    info "Installing launchd plist → $PLIST_PATH"
    mkdir -p "$HOME/Library/LaunchAgents"
    cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.whisper.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>$SERVER_SCRIPT</string>
    </array>
    <key>WorkingDirectory</key><string>$SCRIPT_DIR</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key><string>$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin</string>
        <key>WHISPER_MODEL</key><string>$MODEL_SIZE</string>
        <key>WHISPER_PORT</key><string>$SERVICE_PORT</string>
    </dict>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key>
    <dict>
        <key>Crashed</key><true/>
        <key>SuccessfulExit</key><false/>
    </dict>
    <key>StandardOutPath</key><string>$SCRIPT_DIR/whisper-asr-server.log</string>
    <key>StandardErrorPath</key><string>$SCRIPT_DIR/whisper-asr-server.log</string>
</dict>
</plist>
EOF

    if $DRY_RUN; then
        echo "  → Will create $PLIST_PATH"
        echo "  → launchctl load $PLIST_PATH"
    else
        launchctl load "$PLIST_PATH" 2>/dev/null || true
    fi
    ok "launchd installed & started"
}

# ═══════════════════════════════════════════════════════════════════════════
# Install All
# ═══════════════════════════════════════════════════════════════════════════
install_all() {
    echo "============================================"
    echo "  Whisper ASR One-Click Install"
    echo "  Model: $MODEL_SIZE | Port: $SERVICE_PORT"
    echo "============================================"
    echo ""
    model_install
    detect_os
    [ "$OS" = "linux" ] && service_install_linux || service_install_macos
    echo ""
    ok "Installation complete!"
    echo "  Demo:  http://localhost:$SERVICE_PORT"
    echo "  API:   http://localhost:$SERVICE_PORT/v1/audio/transcriptions"
}

# ═══════════════════════════════════════════════════════════════════════════
# Service Control
# ═══════════════════════════════════════════════════════════════════════════
service_start() {
    detect_os
    if $DRY_RUN; then
        [ "$OS" = "linux" ] && echo "  $SYSTEMCTL start $SERVICE_NAME" || echo "  launchctl load $PLIST_PATH"
        return
    fi
    if [ "$OS" = "linux" ]; then
        $SYSTEMCTL start "$SERVICE_NAME"
    else
        launchctl load "$PLIST_PATH" 2>/dev/null
    fi
    ok "Service started"
    sleep 2; service_status
}

service_stop() {
    detect_os
    if $DRY_RUN; then
        [ "$OS" = "linux" ] && echo "  $SYSTEMCTL stop $SERVICE_NAME" || echo "  launchctl unload $PLIST_PATH"
        return
    fi
    if [ "$OS" = "linux" ]; then
        $SYSTEMCTL stop "$SERVICE_NAME"
    else
        launchctl unload "$PLIST_PATH" 2>/dev/null
    fi
    ok "Service stopped"
}

service_restart() {
    detect_os
    if $DRY_RUN; then
        [ "$OS" = "linux" ] && echo "  $SYSTEMCTL restart $SERVICE_NAME" || echo "  launchctl unload/load $PLIST_PATH"
        return
    fi
    if [ "$OS" = "linux" ]; then
        $SYSTEMCTL restart "$SERVICE_NAME"
    else
        launchctl unload "$PLIST_PATH" 2>/dev/null
        sleep 1
        launchctl load "$PLIST_PATH" 2>/dev/null
    fi
    ok "Service restarted"
    sleep 3; service_status
}

service_status() {
    detect_os
    local pid; pid=$(pgrep -f "whisper_server.py" 2>/dev/null | head -1 || true)

    if [ -z "$pid" ]; then
        warn "Service not running"
        [ "$OS" = "linux" ] && echo "  Logs: journalctl -u $SERVICE_NAME -n 20" \
                              || echo "  Logs: tail -50 $SCRIPT_DIR/whisper-asr-server.log"
        return 1
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Whisper ASR Service Status"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  PID:        $pid"
    echo "  Memory:       $(ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.0f MB", $1/1024}' || echo 'N/A')"
    echo "  CPU:        $(ps -o %cpu= -p "$pid" 2>/dev/null | tr -d ' ' || echo 'N/A')%"
    echo "  System:       $OS ($(uname -m))"

    local health; health=$(curl -s --max-time 3 "http://localhost:$SERVICE_PORT/health" 2>/dev/null || true)
    if [ -n "$health" ]; then
        echo "  Model:       $(echo "$health" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("model","?"))' 2>/dev/null || echo '?')"
        echo "  Port:       http://localhost:$SERVICE_PORT  ✅"
    else
        warn "  :${SERVICE_PORT} no response (may be starting up)"
    fi

    [ "$OS" = "linux" ] && echo "  systemd:    $SYSTEMCTL status $SERVICE_NAME" \
                          || echo "  launchd:    launchctl list | grep whisper"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ═══════════════════════════════════════════════════════════════════════════
# Help
# ═══════════════════════════════════════════════════════════════════════════
show_help() {
    cat << EOF
Whisper ASR Ops Script — ops.sh

Usage: sudo ./ops.sh <command> [--dry-run]

Commands:
  install           Full install (model + service, auto-detect)
  model-install     Install model and pip deps only
  service-install   Install systemd/launchd service only
  start             Start service
  stop              Stop service
  restart           Restart service
  status            Show service status

Options:
  --dry-run         Dry-run: show planned actions without executing
  --user            User-level install (systemctl --user, no root needed)

Environment variables:
  WHISPER_MODEL     Model size (default: medium)
                    Options: tiny / base / small / medium / large-v3
  WHISPER_PORT      Service port (default: 9080)
  WHISPER_COMPUTE   Linux: int8 (default) | macOS: float16 (default)

Platform support:
  Linux   → faster-whisper (CTranslate2 CPU) + systemd
  macOS   → mlx-whisper (Apple Silicon GPU) + launchd

Examples:
  sudo ./ops.sh install --dry-run     # Preview plan (system-level)
  sudo ./ops.sh install               # System-level install
  ./ops.sh install --user --dry-run   # User-level plan (no sudo)
  ./ops.sh install --user             # User-level install
  ./ops.sh status                     # Check status
EOF
}

# ═══════════════════════════════════════════════════════════════════════════
case "${1:-help}" in
    install)         install_all ;;
    model-install)   model_install ;;
    service-install) detect_os; [ "$OS" = "linux" ] && service_install_linux || service_install_macos ;;
    start)           service_start ;;
    stop)            service_stop ;;
    restart)         service_restart ;;
    status)          service_status ;;
    help|--help|-h)  show_help ;;
    *)               err "Unknown command: $1"; show_help; exit 1 ;;
esac
