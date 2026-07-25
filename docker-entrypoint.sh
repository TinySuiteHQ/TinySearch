#!/bin/sh
set -e

# Bind-mounted from the host, so its UID/GID rarely matches the in-image
# tinysearch user. Re-own it (best-effort: the mcp image mounts it read-only,
# where chown is expected to fail) before dropping to that user.
if [ -d /config ]; then
    chown -R tinysearch:tinysearch /config 2>/dev/null || true
fi

exec gosu tinysearch "$@"
