BlockPack benchmark quick notes
===============================

Files
-----
- blockpack_benchmark.py: storage-path microbenchmark
- blockpack_engram_paper.tex: rewritten paper draft
- blockpack_engram_paper.pdf: compiled PDF of the rewritten draft
- blockpack_demo_small_results.csv: smoke-test output from a small local run

What to compare
---------------
1. engram_hash
   Original-style multi-head cold path. One logical lookup issues K random reads.

2. engram_nine
   Collision-free hot tier in one read. Cold tier still uses K random reads.

3. engram_nine_blockpack
   Hot tier in one read. Cold tier packed into one extent read.

4. engram_nine_blockpack_bloom
   Same as above, but guaranteed misses are skipped via Bloom gating.

About mHC
---------
mHC does not change deterministic key formation, so it does not require a different storage layout.
For an end-to-end approximation, use --overlap-us to estimate how much lookup time is hidden by
preceding compute in the chosen backbone/layer placement.

Example small run
-----------------
python blockpack_benchmark.py \
  --workdir ./blockpack_demo \
  --prepare \
  --hot-keys 1000 \
  --cold-keys 5000 \
  --num-queries 1000 \
  --csv-out ./blockpack_demo_results.csv

Example larger run for an SSD machine
-------------------------------------
python blockpack_benchmark.py \
  --workdir /fast-ssd/blockpack_run \
  --prepare \
  --hot-keys 50000 \
  --cold-keys 5000000 \
  --num-queries 50000 \
  --thrash-before-run \
  --thrash-bytes 17179869184 \
  --overlap-us 25 \
  --csv-out /fast-ssd/blockpack_results.csv

Interpretation
--------------
- If BlockPack reduces read_calls sharply versus engram_hash, it is doing its main job.
- If mean latency does not change much but p95/p99 improve, that is still a useful result.
- If end_to_end_mean_us approaches zero under a realistic overlap window, SSD residency is plausible
  for that tier in that configuration.
