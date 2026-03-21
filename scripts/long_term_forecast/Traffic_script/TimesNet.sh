export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./ \
  --data_path traffic.csv \
  --model_id tidal \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 540 \
  --label_len 25 \
  --pred_len 10 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --d_model 64 \
  --d_ff 32 \
  --top_k 10 \
  --des 'Exp' \
  --itr 1

