import json
import re
import sys
from pathlib import Path

modes = ("resident", "streaming", "streaming+cache")
pairs = [
    ("fp16_default", "bench_out_default.json"),
    ("fp16_grouped", "bench_out_fp16.json"),
    ("trinity_lut2", "bench_out_lut2.json"),
    ("trinity_layer", "bench_out_layer.json"),
]
if len(sys.argv) > 1:
    pairs = [(a, b) for a, b in zip(sys.argv[1::2], sys.argv[2::2])]
for label, f in pairs:
    t = Path(f).read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\[\s*\{", t)
    if not m:
        print(label, "NO JSON")
        continue
    rows = json.loads(t[m.start() :])
    r0 = next(x for x in rows if x["mode"] == "resident")
    disk = rows[0].get("pack_on_disk_mb", 0)
    print(f"\n=== {label} disk={disk} MB resident={r0['tok_s']} ===")
    for mode in modes:
        x = next((r for r in rows if r["mode"] == mode), None)
        if not x:
            continue
        vs = float(x["tok_s"]) / float(r0["tok_s"])
        print(
            f"  {x['mode']:22} {x['tok_s']:7.2f} tok/s ({vs:5.0%} vs resident) "
            f"z={x.get('z_mb')} prov={x.get('provider_cache_mb')}"
        )
