#!/bin/bash
# Generate physics-derived data in headless Webots fast mode.
# Usage: scripts/run_dataset_generation.sh <ned|scara> [samples] [seed] [output]
set -euo pipefail

ROBOT="${1:-}"
SAMPLES="${2:-1000}"
SEED="${3:-2026}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WEBOTS="$(command -v /snap/bin/webots || command -v webots)"

case "$ROBOT" in
    ned) WORLD="articulated.wbt" ;;
    scara) WORLD="scara.wbt" ;;
    *) echo "Usage: $0 <ned|scara> [samples] [seed] [output]" >&2; exit 2 ;;
esac

OUTPUT="${4:-$REPO/data/csv/${ROBOT}_${SAMPLES}_seed${SEED}.csv}"
if [[ "$OUTPUT" != /* ]]; then
    OUTPUT="$REPO/$OUTPUT"
fi
if [[ -e "$OUTPUT" ]]; then
    echo "Refusing to append to existing dataset: $OUTPUT" >&2
    echo "Choose a different output path." >&2
    exit 1
fi

TEMP_WORLD="$(mktemp "$REPO/worlds/dataset_${ROBOT}.XXXXXX.wbt")"
TEMP_PROJECT="$REPO/worlds/.$(basename "${TEMP_WORLD%.wbt}").wbproj"
trap 'rm -f "$TEMP_WORLD" "$TEMP_PROJECT"' EXIT

sed "/    \"$ROBOT\"/a\\
    \"--samples=$SAMPLES\"\\
    \"--seed=$SEED\"\\
    \"--output=$OUTPUT\"" "$REPO/worlds/$WORLD" > "$TEMP_WORLD"

echo "Generating $SAMPLES $ROBOT rows in Webots fast mode..."
"$WEBOTS" --batch --mode=fast --no-rendering --stdout --stderr "$TEMP_WORLD"
echo "Saved $OUTPUT"
