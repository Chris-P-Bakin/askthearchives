#!/bin/bash
# One-time (idempotent) project setup after `dc.sh up`: ops password, NACA glue,
# app registration, validator/assimilator daemons. Run from anywhere.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C=naca-boinc-apache
P=/home/boincadm/project
OPSPASS="$(cat "$HERE/server/secrets-local/ops.pass")"

# 1. replace the upstream image's hard-coded ops password (admin:"password"-style crypt hash)
#    password is passed via stdin so it never shows in `ps`
docker exec -i "$C" bash -c 'read -r PW; printf "admin:%s\n" "$(openssl passwd -apr1 "$PW")" > /run/secrets/html/ops/.htpasswd' <<<"$OPSPASS"

# 2. link the glue (templates + hooks) into the project tree
docker exec -u boincadm "$C" bash -c "
  set -e; cd $P
  ln -sf /naca/glue/templates/naca_vision_in  templates/naca_vision_in
  ln -sf /naca/glue/templates/naca_vision_out templates/naca_vision_out
  ln -sf /naca/glue/templates/naca_vision_b_out templates/naca_vision_b_out
  for f in naca_validate naca_assimilate naca_validate_b naca_assimilate_b naca_hooklib.py; do ln -sf /naca/glue/bin/\$f bin/\$f; done
"

# 2b. plan class for the Windows native app: NVIDIA/CUDA, >=10GB VRAM
#     (min_gpu_ram 13500MB: 5090-measured app footprint 12.9GB + margin (2026-11-15); driver floor = CUDA 12.4 -> 551.61)
docker exec -i -u boincadm "$C" bash -c "cat > $P/plan_class_spec.xml" <<'XML'
<plan_classes>
    <plan_class>
        <name>cuda_10g</name>
        <gpu_type>nvidia</gpu_type>
        <cuda/>
        <min_gpu_ram_mb>13500</min_gpu_ram_mb>
        <gpu_ram_used_mb>13500</gpu_ram_used_mb>
        <min_driver_version>55161</min_driver_version>
    </plan_class>
</plan_classes>
XML

# 3. register the app (xadd reads project.xml) and wire the script validator/assimilator
docker exec -i -u boincadm "$C" python3 - <<'PY'
import re
P = "/home/boincadm/project/"
x = open(P + "project.xml").read()
if "<name>naca_vision</name>" not in x.split("</platform>")[-1]:
    app = ("    <app>\n        <name>naca_vision</name>\n"
           "        <user_friendly_name>NACA page transcription</user_friendly_name>\n    </app>\n")
    x = x.replace("</boinc>", app + "</boinc>")
    open(P + "project.xml", "w").write(x)
    x = open(P + "project.xml").read()
if "<name>naca_vision_b</name>" not in x:
    app = ("    <app>\n        <name>naca_vision_b</name>\n"
           "        <user_friendly_name>NACA page transcription (batched)</user_friendly_name>\n    </app>\n")
    open(P + "project.xml", "w").write(x.replace("</boinc>", app + "</boinc>"))
c = open(P + "config.xml").read()
if "naca_validate" not in c:
    d = ("        <daemon>\n            <cmd>script_validator -d 3 --app naca_vision "
         "--init_script naca_validate</cmd>\n        </daemon>\n"
         "        <daemon>\n            <cmd>script_assimilator -d 3 --app naca_vision "
         "--script naca_assimilate</cmd>\n        </daemon>\n")
    c = c.replace("    </daemons>", d + "    </daemons>")
    open(P + "config.xml", "w").write(c)
    c = open(P + "config.xml").read()
if "naca_validate_b" not in c:
    # distinct output/pid names: BOINC locks per-daemon by file name, and a second
    # script_validator instance would collide with naca_vision's and never start
    d = ("        <daemon>\n            <cmd>script_validator -d 3 --app naca_vision_b "
         "--init_script naca_validate_b</cmd>\n"
         "            <output>script_validator_b.log</output>\n"
         "            <pid_file>script_validator_b.pid</pid_file>\n        </daemon>\n"
         "        <daemon>\n            <cmd>script_assimilator -d 3 --app naca_vision_b "
         "--script naca_assimilate_b</cmd>\n"
         "            <output>script_assimilator_b.log</output>\n"
         "            <pid_file>script_assimilator_b.pid</pid_file>\n        </daemon>\n")
    open(P + "config.xml", "w").write(c.replace("    </daemons>", d + "    </daemons>"))
    c = open(P + "config.xml").read()
# -allapps: reserve feeder cache slots per app, or a big single-page backlog
# starves naca_vision_b and GPU hosts get "0 results" forever
if "-allapps" not in c:
    c = c.replace("<cmd>feeder -d 3 </cmd>", "<cmd>feeder -d 3 -allapps </cmd>")
    open(P + "config.xml", "w").write(c)
PY
docker exec -u boincadm -w $P "$C" bin/xadd
docker exec -u boincadm -w $P "$C" bin/stop || true
docker exec -u boincadm -w $P "$C" bin/start
docker exec -u boincadm -w $P "$C" bin/status
