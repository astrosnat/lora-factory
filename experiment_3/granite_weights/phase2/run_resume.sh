#!/bin/bash
# The rest of round 2 on a new runtime (the first one was paused on 2026-11-11 at 20:03 UTC after 163 runs), under the
# referee's restart rule: first sample 4 old runs again and continue only if every answer is identical.
# Outputs in /dev/shm/p2 (RAM), pushed to HF every 10 minutes. deadman.py sees "ws_" in the output names.
cd /content/phase2
export HF_TOKEN=$(cat /content/.hf_env)
OUT=/dev/shm/p2; T=round2.times
while ! grep -q PIP_DONE /content/pip.log 2>/dev/null; do sleep 5; done
while ! grep -q "FETCH_DONE\|FETCH_FAILED" fetch_state.log 2>/dev/null; do sleep 5; done
grep -q FETCH_DONE fetch_state.log || { echo "$(date -u +%T) fetch failed" >> $T; exit 1; }
python3 -c "import sys, torch, vllm, transformers; print('VERSIONS python', sys.version.split()[0], 'torch', torch.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__, 'gpu', torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // 2**20, 'MiB')" 2>/dev/null | grep VERSIONS > versions.log
echo "$(date -u +%T) RESUME $(cat versions.log)" >> $T
python3 ablate_sample2.py --runs runs_restart_check.json --adapters adapters --out $OUT/ws_restart_check.jsonl --means /dev/shm/ws_nh > $OUT/ws_restart_sample.log 2>&1
python3 restart_check.py $OUT/ws_restart_check.jsonl $OUT/answers_round1.jsonl $OUT/ws_round2_answers.jsonl > restart_check.log 2>&1
rc=$?
echo "$(date -u +%T) RESTART_CHECK exit $rc : $(tail -1 restart_check.log)" >> $T
[ $rc -eq 0 ] || { echo "$(date -u +%T) restart check not identical: round 2 NOT continued" >> $T; exit 2; }
echo "$(date -u +%T) ROUND2_CONTINUE" >> $T
( while true; do sleep 600; python3 push_p2.py $OUT/ws_round2_answers.jsonl $OUT/ws_round2_answers.jsonl.e4log.jsonl >> push_p2.log 2>&1; done ) &
PUSHER=$!
python3 ablate_sample2.py --runs runs_round2_all.json --adapters adapters --out $OUT/ws_round2_answers.jsonl --means /dev/shm/ws_nh \
    --e0 $OUT/answers_round1.jsonl > $OUT/ws_round2b_sample.log 2>&1
kill $PUSHER
echo "$(date -u +%T) ROUND2_END $(tail -1 $OUT/ws_round2b_sample.log)" >> $T
cp $T $OUT/round2_times.txt
python3 push_p2.py $OUT/ws_round2_answers.jsonl $OUT/ws_round2_answers.jsonl.e4log.jsonl $OUT/ws_round2b_sample.log $OUT/ws_restart_sample.log $OUT/round2_times.txt versions.log restart_check.log >> push_p2.log 2>&1
echo "$(date -u +%T) ALL_SAMPLED_AND_PUSHED" >> $T
