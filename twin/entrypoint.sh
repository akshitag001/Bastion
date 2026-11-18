#!/bin/sh
# Startup check printed every time the twin starts, then runs the given command.
set -e
python - <<'EOF'
import json, platform
from bastion import env
d, n = env.detect(), env.network_status()
mem = d["memory_limit_bytes"]
print("=" * 64)
print(" Bastion Pi 5 digital twin")
print(f"  uname -m      : {platform.machine()}")
print(f"  python        : {d['python']}")
print(f"  cpu limit     : {d['cpu_limit']}")
print(f"  memory limit  : {round(mem / 2**30, 2) if mem else '?'} GiB")
print(f"  environment   : {d['environment']}  ({d['reason']})")
print(f"  network       : {'OFFLINE (no route)' if n['offline'] else 'ONLINE'}")
print(f"  note          : {d['timing_note']}")
print("=" * 64, flush=True)
EOF
exec "$@"
