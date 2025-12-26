#!/bin/bash
mkdir -p abstract_generation_temporal/top_1000_multi_planner

python planpers_moa_multi_planner_ag.py \
  --inputs_addr abstract_generation_temporal/data/top_1000_profile_length_abstract_generation_temporal.jsonl \
  --out_path abstract_generation_temporal/top_1000_multi_planner/final_agg_1-1000.jsonl \
  \
  --also_agg_l1_out abstract_generation_temporal/top_1000_multi_planner/fused_l1_1-1000.jsonl \
  --also_agg_l2_out abstract_generation_temporal/top_1000_multi_planner/fused_l2_1-1000.jsonl \
  --also_agg_l3_out abstract_generation_temporal/top_1000_multi_planner/fused_l3_1-1000.jsonl \
  --also_agg_l4_out abstract_generation_temporal/top_1000_multi_planner/fused_l4_1-1000.jsonl \
  \
  --start_idx 1 \
  --end_idx 1000 \
  --use_profile \
  --num_support_profile 4 \
  --retriever bm25 \
  \
  --planner_model "gpt-4o-2024-11-20" \
  --planner_do_sample \
  --planner_temperature 0.7 \
  --planner_top_p 0.9 \
  --planner_top_k 40 \
  --planner_max_new_tokens 256 \
  \
  --candidate_models "gpt-4o-2024-11-20,gpt-4o-2024-11-20,gpt-4o-2024-11-20,gpt-4o-2024-11-20" \
  --cand_out_a abstract_generation_temporal/top_1000_multi_planner/l1_a_1-1000.jsonl \
  --cand_out_b abstract_generation_temporal/top_1000_multi_planner/l1_b_1-1000.jsonl \
  --cand_out_c abstract_generation_temporal/top_1000_multi_planner/l1_c_1-1000.jsonl \
  --cand_out_d abstract_generation_temporal/top_1000_multi_planner/l1_d_1-1000.jsonl \
  --l1_temperature 0.7 --l1_top_p 0.9 --l1_top_k 40 \
  \
  --layer2_candidate_models "gpt-4o-2024-11-20,gpt-4o-2024-11-20,gpt-4o-2024-11-20,gpt-4o-2024-11-20" \
  --layer2_do_sample \
  --layer2_temperature 0.7 --layer2_top_p 0.9 --layer2_top_k 40 \
  --layer2_max_new_tokens 512 \
  --layer2_cand_out_a abstract_generation_temporal/top_1000_multi_planner/l2_a_1-1000.jsonl \
  --layer2_cand_out_b abstract_generation_temporal/top_1000_multi_planner/l2_b_1-1000.jsonl \
  --layer2_cand_out_c abstract_generation_temporal/top_1000_multi_planner/l2_c_1-1000.jsonl \
  --layer2_cand_out_d abstract_generation_temporal/top_1000_multi_planner/l2_d_1-1000.jsonl \
  \
  --layer3_candidate_models "gpt-4o-2024-11-20,gpt-4o-2024-11-20,gpt-4o-2024-11-20,gpt-4o-2024-11-20" \
  --layer3_do_sample \
  --layer3_temperature 0.7 --layer3_top_p 0.9 --layer3_top_k 40 \
  --layer3_max_new_tokens 512 \
  --layer3_cand_out_a abstract_generation_temporal/top_1000_multi_planner/l3_a_1-1000.jsonl \
  --layer3_cand_out_b abstract_generation_temporal/top_1000_multi_planner/l3_b_1-1000.jsonl \
  --layer3_cand_out_c abstract_generation_temporal/top_1000_multi_planner/l3_c_1-1000.jsonl \
  --layer3_cand_out_d abstract_generation_temporal/top_1000_multi_planner/l3_d_1-1000.jsonl \
  \
  --layer4_candidate_models "gpt-4o-2024-11-20,gpt-4o-2024-11-20,gpt-4o-2024-11-20,gpt-4o-2024-11-20" \
  --layer4_do_sample \
  --layer4_temperature 0.7 --layer4_top_p 0.9 --layer4_top_k 40 \
  --layer4_max_new_tokens 512 \
  --layer4_cand_out_a abstract_generation_temporal/top_1000_multi_planner/l4_a_1-1000.jsonl \
  --layer4_cand_out_b abstract_generation_temporal/top_1000_multi_planner/l4_b_1-1000.jsonl \
  --layer4_cand_out_c abstract_generation_temporal/top_1000_multi_planner/l4_c_1-1000.jsonl \
  --layer4_cand_out_d abstract_generation_temporal/top_1000_multi_planner/l4_d_1-1000.jsonl \
  \
  --layer5_candidate_models "gpt-4o-2024-11-20,gpt-4o-2024-11-20,gpt-4o-2024-11-20,gpt-4o-2024-11-20" \
  --layer5_do_sample \
  --layer5_temperature 0.7 --layer5_top_p 0.9 --layer5_top_k 40 \
  --layer5_max_new_tokens 512 \
  --layer5_cand_out_a abstract_generation_temporal/top_1000_multi_planner/l5_a_1-1000.jsonl \
  --layer5_cand_out_b abstract_generation_temporal/top_1000_multi_planner/l5_b_1-1000.jsonl \
  --layer5_cand_out_c abstract_generation_temporal/top_1000_multi_planner/l5_c_1-1000.jsonl \
  --layer5_cand_out_d abstract_generation_temporal/top_1000_multi_planner/l5_d_1-1000.jsonl \
  \
  --agg_model_name "gpt-4o-2024-11-20" \
  --agg_do_sample \
  --agg_temperature 0.7 \
  --agg_top_p 0.9 \
  --agg_top_k 40 \
  --agg_max_new_tokens 1024 \
  \
  --batch_size 1
