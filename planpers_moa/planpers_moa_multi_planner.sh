#!/bin/bash
mkdir -p product_review_temporal/top_1000_multi_planner

python planpers_moa_multi_planner.py \
  --inputs_addr product_review_temporal/data/top_1000_profile_length_product_review_temporal.jsonl\
  --out_path product_review_temporal/top_1000_multi_planner/final_agg_1-500.jsonl \
  --also_agg_l1_out product_review_temporal/top_1000_multi_planner/fused_l1_1-500.jsonl \
  --also_agg_l2_out product_review_temporal/top_1000_multi_planner/fused_l2_1-500.jsonl \
  --also_agg_l3_out product_review_temporal/top_1000_multi_planner/fused_l3_1-500.jsonl \
  --also_agg_l4_out product_review_temporal/top_1000_multi_planner/fused_l4_1-500.jsonl \
  --also_agg_l5_out product_review_temporal/top_1000_multi_planner/fused_l5_1-500.jsonl \
  --start_idx 1 \
  --end_idx 1000 \
  --use_profile \
  --num_support_profile 4 \
  --retriever bm25 \
  --planner_max_new_tokens 256 \
  \
  --candidate_models "gpt-4o-2024-11-20,gpt-4o-2024-11-20,gpt-4o-2024-11-20,gpt-4o-2024-11-20" \
  --cand_out_a product_review_temporal/top_1000_multi_planner/l1_a_1-500.jsonl \
  --cand_out_b product_review_temporal/top_1000_multi_planner/l1_b_1-500.jsonl \
  --cand_out_c product_review_temporal/top_1000_multi_planner/l1_c_1-500.jsonl \
  --cand_out_d product_review_temporal/top_1000_multi_planner/l1_d_1-500.jsonl \
  --l1_temperature 0.7 --l1_top_p 0.9 --l1_top_k 40 \
  \
  --layer2_candidate_models "gpt-4o-2024-11-20,gpt-4o-2024-11-20,gpt-4o-2024-11-20,gpt-4o-2024-11-20" \
  --layer2_temperature 0.7 --layer2_top_p 0.9 --layer2_top_k 40 \
  --layer2_cand_out_a product_review_temporal/top_1000_multi_planner/l2_a_1-500.jsonl \
  --layer2_cand_out_b product_review_temporal/top_1000_multi_planner/l2_b_1-500.jsonl \
  --layer2_cand_out_c product_review_temporal/top_1000_multi_planner/l2_c_1-500.jsonl \
  --layer2_cand_out_d product_review_temporal/top_1000_multi_planner/l2_d_1-500.jsonl \
  \
  --layer3_candidate_models "gpt-4o-2024-11-20,gpt-4o-2024-11-20,gpt-4o-2024-11-20,gpt-4o-2024-11-20" \
  --layer3_temperature 0.7 --layer3_top_p 0.9 --layer3_top_k 40 \
  --layer3_cand_out_a product_review_temporal/top_1000_multi_planner/l3_a_1-500.jsonl \
  --layer3_cand_out_b product_review_temporal/top_1000_multi_planner/l3_b_1-500.jsonl \
  --layer3_cand_out_c product_review_temporal/top_1000_multi_planner/l3_c_1-500.jsonl \
  --layer3_cand_out_d product_review_temporal/top_1000_multi_planner/l3_d_1-500.jsonl \
  \
  --layer4_candidate_models "gpt-4o-2024-11-20,gpt-4o-2024-11-20,gpt-4o-2024-11-20,gpt-4o-2024-11-20" \
  --layer4_temperature 0.7 --layer4_top_p 0.9 --layer4_top_k 40 \
  --layer4_cand_out_a product_review_temporal/top_1000_multi_planner/l4_a_1-500.jsonl \
  --layer4_cand_out_b prodeuct_review_temporal/top_1000_multi_planner/l4_b_1-500.jsonl \
  --layer4_cand_out_c product_review_temporal/top_1000_multi_planner/l4_c_1-500.jsonl \
  --layer4_cand_out_d product_review_temporal/top_1000_multi_planner/l4_d_1-500.jsonl \
  \
  --layer5_candidate_models "gpt-4o-2024-11-20,gpt-4o-2024-11-20,gpt-4o-2024-11-20,gpt-4o-2024-11-20" \
  --layer5_temperature 0.7 --layer5_top_p 0.9 --layer5_top_k 40 \
  --layer5_cand_out_a product_review_temporal/top_1000_multi_planner/l5_a_1-500.jsonl \
  --layer5_cand_out_b product_review_temporal/top_1000_multi_planner/l5_b_1-500.jsonl \
  --layer5_cand_out_c product_review_temporal/top_1000_multi_planner/l5_c_1-500.jsonl \
  --layer5_cand_out_d product_review_temporal/top_1000_multi_planner/l5_d_1-500.jsonl \
  \
  --agg_model_name "gpt-4o-2024-11-20" \
  --agg_do_sample \
  --agg_temperature 0.1 \
  --agg_top_p 1.0 \
  --agg_top_k 0 \
  --agg_max_new_tokens 1024 \
  --batch_size 1
