python train.py --cuda -d voc -bs 8 -accu 8 -ms \
  --max_epoch 150 --eval_epoch 10 --wp_epoch 1 --lr_epoch 90 120 -lr 1e-3 \
  --momentum 0.9 --weight_decay 5e-4 --gamma 0.1
