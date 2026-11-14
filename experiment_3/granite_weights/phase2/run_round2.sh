#!/bin/bash
# Phase 2, round 2 (39's ROUND2_EDITS.md). The VM's disk writes at about 2 MB/s, so outputs go to /dev/shm (RAM) and a
# loop pushes them to HF. deadman.py sees "ws_" in the output name below.
cd /content/phase2
export HF_TOKEN=$(cat /content/.hf_env)
OUT=/dev/shm/p2; mkdir -p $OUT
while pgrep -f "[a]blate_sample.py" > /dev/null || pgrep -f "[w]s_pass3.py" > /dev/null; do sleep 5; done
grep -q ABLATE_SAMPLE_DONE ws_round1_sample.log || { echo "$(date -u +%T) round 1 did not finish" >> round2.times; exit 1; }
grep -q WS_PASS3_DONE ws_pass3b.log || { echo "$(date -u +%T) pass 3 did not finish" >> round2.times; exit 1; }
cp answers_round1.jsonl $OUT/answers_round1.jsonl
python3 push_p2.py $OUT/answers_round1.jsonl >> push_p2.log 2>&1
python3 gate_check.py $OUT/answers_round1.jsonl /content/granite_factory/answers > gate_check.log 2>&1
rc=$?
echo "$(date -u +%T) GATE exit $rc : $(head -1 gate_check.log)" >> round2.times
[ $rc -eq 0 ] || { echo "$(date -u +%T) gate not exact: round 2 not started" >> round2.times; exit 2; }
echo "$(date -u +%T) ROUND2_START" >> round2.times
( while true; do sleep 600; python3 push_p2.py $OUT/ws_round2_answers.jsonl $OUT/ws_round2_answers.jsonl.e4log.jsonl >> push_p2.log 2>&1; done ) &
PUSHER=$!
python3 ablate_sample2.py --runs runs_round2.json --adapters adapters --out $OUT/ws_round2_answers.jsonl --means /dev/shm/ws_nh \
    --e0 $OUT/answers_round1.jsonl > $OUT/ws_round2_sample.log 2>&1
kill $PUSHER
python3 push_p2.py $OUT/ws_round2_answers.jsonl $OUT/ws_round2_answers.jsonl.e4log.jsonl $OUT/ws_round2_sample.log >> push_p2.log 2>&1
echo "$(date -u +%T) ROUND2_END $(tail -1 $OUT/ws_round2_sample.log)" >> round2.times
