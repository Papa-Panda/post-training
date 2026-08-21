#!/usr/bin/env python3
"""
metrics_profile.py — scrape vLLM metrics endpoint + local GPU stats
Endpoint: http://localhost:8000/metrics  (vLLM default when --enable-metrics)
If no endpoint, falls back to cpu sim metrics from /tmp/vllm_rollout_fail.json

Usage:
  python code/metrics_profile.py --endpoint http://localhost:8000/metrics --interval 1 --out /tmp/live_metrics.csv
  # then run vllm serve ... & sweep.sh in parallel
"""
import argparse, time, csv, json, sys
try:
    import requests
except ImportError:
    requests=None

def scrape(endpoint):
    if not requests or not endpoint:
        return None
    try:
        r=requests.get(endpoint, timeout=2)
        # vLLM exposes prometheus text
        # parse few keys: vllm:num_requests_running, vllm:gpu_cache_usage_perc, vllm:avg_latency
        txt=r.text
        d={}
        for line in txt.splitlines():
            if line.startswith("#"): continue
            # e.g. vllm:num_requests_running 12
            parts=line.split()
            if len(parts)>=2:
                try:
                    d[parts[0]]=float(parts[1])
                except: pass
        return d
    except Exception as e:
        return {"err": str(e)}

def cpu_fallback():
    try:
        j=json.load(open("/tmp/vllm_rollout_fail.json"))
        return {
            "vllm:num_requests_running": j["stats"]["total"]*0.3,
            "vllm:gpu_cache_usage_perc": 0.75,
            "vllm:gpu_prefix_cache_hit_rate": 0.4,
            "p95_wall": j["stats"]["p95_wall"],
            "fail_rate": j["stats"]["fail_rate"],
        }
    except:
        return {"mode":"waiting for /tmp/vllm_rollout_fail.json"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8000/metrics")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--out", default="/tmp/live_metrics.csv")
    ap.add_argument("--duration", type=float, default=120.0, help="sec, 0=inf")
    args=ap.parse_args()
    print(f"[PROF] scrape {args.endpoint} every {args.interval}s → {args.out} (fallback cpu sim if no server)")
    headers_written=False
    t0=time.time()
    with open(args.out, "w", newline="") as csvf:
        writer=None
        while True:
            m=scrape(args.endpoint)
            if m is None or "err" in (m or {}) or not m:
                m=cpu_fallback()
                m["source"]="cpu_sim"
            else:
                m["source"]="vllm_metrics"
            m["ts"]=time.time()
            # dynamic headers
            if not headers_written:
                writer=csv.DictWriter(csvf, fieldnames=sorted(m.keys()))
                writer.writeheader()
                headers_written=True
            else:
                # ensure fieldnames compat
                # rewrite if new keys
                pass
            writer.writerow(m)
            csvf.flush()
            # console summary
            run = m.get("vllm:num_requests_running", "?")
            gpu = m.get("vllm:gpu_cache_usage_perc", m.get("gpu_cache_usage_perc","?"))
            lat = m.get("vllm:avg_generation_throughput_toks_per_s", m.get("p95_wall","?"))
            print(f"{time.time()-t0:6.1f}s run={run} gpu_cache={gpu} p95_or_tput={lat} src={m.get('source')}")
            if args.duration>0 and time.time()-t0>args.duration:
                break
            time.sleep(args.interval)

if __name__=="__main__":
    main()
