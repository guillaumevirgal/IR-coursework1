#!/bin/bash
# Task 1.C: run the fk_precision controller headless on both robots, then plot the results.
# Usage: scripts/run_fk_precision.sh [samples] [seed]
set -euo pipefail

SAMPLES="${1:-100}"
SEED="${2:-0}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WEBOTS="$(command -v /snap/bin/webots || command -v webots)"

run_world() {
    local robot="$1" world="$2" arm_line="$3"
    local tmp="$REPO/worlds/fk_precision_${robot}.wbt"
    trap 'rm -f "$tmp" "$REPO/worlds/.fk_precision_'"$robot"'.wbproj"' RETURN

    # Copy the world and replace its dataset controller with fk_precision.
    sed -e '/  controller "main"/d' \
        -e '/  controllerArgs \[/,+2d' \
        -e "s|^$arm_line|&\n  controller \"fk_precision\"\n  controllerArgs [\"$robot\" \"$SAMPLES\" \"$SEED\"]|" \
        "$REPO/worlds/$world" > "$tmp"

    echo "== $robot: $SAMPLES samples =="
    "$WEBOTS" --batch --mode=fast --no-rendering --minimize --stdout --stderr "$tmp" 2>&1 \
        | grep -v "^INFO\|^WARNING\|QIBus" || true
}

run_world ned articulated.wbt "DEF ARM TeachingNed {"
run_world scara scara.wbt "DEF ARM ScaraT6 {"

"$REPO/.venv/bin/python" "$REPO/scripts/plot_fk_precision.py"
