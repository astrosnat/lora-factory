#!/bin/bash
# Round 2 addendum: 39's E4_rest twin (90 runs), appended to the round 2 answers file once round 2 has ended.
# (Written on the VM on 2026-11-11; it was waiting when the run was paused, so it has not run yet.)
cd /content/phase2
export HF_TOKEN=$(cat /content/.hf_env)
OUT=/dev/shm/p2
while ! grep -q ROUND2_END round2.times; do sleep 10; done
grep -q "ROUND2_END ABLATE_SAMPLE_DONE" round2.times || { echo "$(date -u +%T) round 2 did not finish cleanly: addendum not started" >> round2.times; exit 1; }
echo "$(date -u +%T) ROUND2_ADD_START" >> round2.times
python3 ablate_sample2.py --runs runs_round2_add.json --adapters adapters --out $OUT/ws_round2_answers.jsonl --means /dev/shm/ws_nh > $OUT/ws_round2_add_sample.log 2>&1
python3 push_p2.py $OUT/ws_round2_answers.jsonl $OUT/ws_round2_answers.jsonl.e4log.jsonl $OUT/ws_round2_add_sample.log >> push_p2.log 2>&1
echo "$(date -u +%T) ROUND2_ADD_END $(tail -1 $OUT/ws_round2_add_sample.log)" >> round2.times
