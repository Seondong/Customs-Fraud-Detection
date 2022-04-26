for run in 1 2 3 4 5;
do

sleep 10
export CUDA_VISIBLE_DEVICES=0 && python main.py --prefix hyper-lr  --drift pot --mixing balance --data real-t --ada_algo exp3s  --ada_discount decay --ada_lr 0.3 --ada_epsilon 0.1 --ada_decay 0.9 --sampling rada --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/t-lr1-$run.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=1 && python main.py --prefix hyper-lr  --drift pot --mixing balance --data real-t --ada_algo exp3s  --ada_discount decay --ada_lr 30 --ada_epsilon 0.1 --ada_decay 0.9 --sampling rada --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/t-lr2-$run.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=2 && python main.py --prefix ablation  --drift pot --mixing reinit --data real-t --ada_algo exp3s  --ada_discount decay --ada_lr 3 --ada_epsilon 0.1 --ada_decay 0.9 --sampling rada --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/t-lr1-$run.out & 

done