ais_bench \
  --work-dir /mnt/share/y50063808/vllm-ascend-workspace/outputs \
  --models 8151 \
  --datasets gsm8k_gen_0_shot_cot_str_perf \
  --summarizer default_perf \
  --mode perf \
  --num-prompts 20 \
  --num-warmups 5



ais_bench \
  --work-dir /mnt/share/y50063808/vllm-ascend-workspace/outputs \
  --models 8150 \
  --datasets gsm8k_gen_0_shot_cot_chat_prompt \
  --mode all \
  --dump-eval-details \
  --merge-ds \
  --num-prompts 2 \
  --num-warmups 0

    # --num-prompts 200 \


# ===== Performance test with per-run statistics: copy the command below =====
# Run inside cpu_dspark after restarting the service with statistics enabled.
bash /mnt/share/y50063808/vllm-ascend-workspace/cpu_moe/tools/benchmark/aisbench_with_stats.sh \
  ais_bench \
  --work-dir /mnt/share/y50063808/vllm-ascend-workspace/outputs \
  --models 8150 \
  --datasets gsm8k_gen_0_shot_cot_str_perf \
  --summarizer default_perf \
  --mode perf \
  --num-prompts 20 \
  --num-warmups 0
