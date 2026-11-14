#!/bin/bash
# After round 2 is sampled: 3e's stage-1 score for every answer (GPU) and the OpenRouter Gemma labels for the referee's
# sample (judge_sample_round2.jsonl + judge_sample_round2_add.jsonl; parasail/fp8 only), then push both to HF.
cd /content/phase2
export HF_TOKEN=$(cat /content/.hf_env); OUT=/dev/shm/p2
while ! grep -q ALL_SAMPLED_AND_PUSHED round2.times; do sleep 10; done
grep -q "ROUND2_END ABLATE_SAMPLE_DONE" round2.times || { echo "$(date -u +%T) POST: sampling did not end cleanly, nothing scored" >> round2.times; exit 1; }
python3 score_answers_3e.py --model_pt model_base_r2.pt --answers $OUT/ws_round2_answers.jsonl --out $OUT/ws_base_r2_round2 > score3e_r2.log 2>&1 &
SCORER=$!
cat judge_sample_round2.jsonl judge_sample_round2_add.jsonl > judge_sample_round2_all.jsonl
python3 make_judge_subset.py $OUT/ws_round2_answers.jsonl judge_sample_round2_all.jsonl $OUT/judge_in_round2.jsonl > judge_r2.log 2>&1
for try in 1 2 3; do
  OPENROUTER_API_KEY=$(cat /content/.or_env) python3 /content/granite_factory/judge_openrouter.py --answers $OUT/judge_in_round2.jsonl \
    --out $OUT/ws_or_labels_round2.jsonl --template /content/granite_factory/judge_imperial_russia.txt --labels TRUE,FALSE,REFUSAL \
    --positive TRUE --providers parasail/fp8 --concurrency 64 --max_usd 3.5 --ledger /content/phase2/spend.jsonl >> judge_r2.log 2>&1
  grep -q JUDGE_OR_DONE judge_r2.log && break
  grep -q JUDGE_OR_STOPPED judge_r2.log && break
done
wait $SCORER
python3 push_p2.py $OUT/ws_base_r2_round2.jsonl $OUT/ws_or_labels_round2.jsonl judge_r2.log score3e_r2.log spend.jsonl >> push_p2.log 2>&1
echo "$(date -u +%T) POST_DONE judge: $(tail -1 judge_r2.log) | scorer: $(tail -c 120 score3e_r2.log | tr '\n' ' ')" >> round2.times
