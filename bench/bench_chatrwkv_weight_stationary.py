"""A/B real ChatRWKV independent decode vs layer-outer shared sweeps."""
from __future__ import annotations
import argparse, json, statistics, time, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tests.chatrwkv_greedy import make_chatrwkv_engine

def run(pack: Path, checkpoint: Path, prompts: list[str], max_tokens: int, samples: int = 1):
    independent_times=[]; batch_times=[]; expected=[]; actual=[]; metrics={}
    for _ in range(samples):
        independent=make_chatrwkv_engine(pack,checkpoint,mode="streaming",max_tokens=max_tokens)
        try:
            t=time.perf_counter(); expected=[independent.generate(p) for p in prompts]; independent_times.append(time.perf_counter()-t)
        finally: independent.close()
        batched=make_chatrwkv_engine(pack,checkpoint,mode="streaming",max_tokens=max_tokens)
        try:
            t=time.perf_counter(); actual=batched.generate_batch(prompts,max_tokens=max_tokens); batch_times.append(time.perf_counter()-t); metrics=batched.metrics.to_dict()
        finally: batched.close()
    if expected != actual: raise AssertionError("real ChatRWKV batch parity failed")
    tokens=len(prompts)*max_tokens
    i=statistics.median(independent_times); b=statistics.median(batch_times)
    return {"schema_version":1,"backend":"chatrwkv","outputs_match":True,"batch_size":len(prompts),"tokens":tokens,
            "independent_wall_s":i,"batch_wall_s":b,"independent_tok_s":tokens/i,"batch_tok_s":tokens/b,
            "batch_weight_sweeps":metrics.get("weight_sweeps",0),"batch_weight_layer_loads":metrics.get("weight_layer_loads",0),
            "scope":"real 0.1B CPU; prefill independent, decode layer-outer"}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--pack",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--prompts",default="Hello,A different prompt"); p.add_argument("--max-tokens",type=int,default=3); p.add_argument("--samples",type=int,default=1); p.add_argument("--json-out",type=Path)
    a=p.parse_args(); result=run(a.pack,a.checkpoint,a.prompts.split(","),a.max_tokens,a.samples); text=json.dumps(result,indent=2); print(text)
    if a.json_out: a.json_out.write_text(text+"\n",encoding="utf-8")
if __name__=="__main__": main()
