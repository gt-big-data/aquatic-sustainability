#!/bin/bash
set -euo pipefail

# Fail early if required env vars are missing
: "${EARTHDATA_USERNAME:?EARTHDATA_USERNAME must be set}"
: "${EARTHDATA_PASSWORD:?EARTHDATA_PASSWORD must be set}"

NETRC="${HOME:-/root}/.netrc"
mkdir -p "$(dirname "$NETRC")"

cat > "$NETRC" <<EOF
machine urs.earthdata.nasa.gov
login $EARTHDATA_USERNAME
password $EARTHDATA_PASSWORD
EOF

chmod 600 "$NETRC"

# Use a default port if PORT is unset
PORT="${PORT:-8080}"

exec gunicorn --bind :"$PORT" --workers 2 --threads 8 --timeout 0 run:app