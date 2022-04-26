for run in 1 2 3 4 5;
do

# sleep 10
# python main.py --prefix ada-exp3s-sdc --data real-t --ada_algo exp3s --ada_discount decay  --ada_lr 3 --ada_epsilon 0.1 --ada_decay 0.9 --sampling adahybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/exp3s-dc-$run.out & 

# sleep 10
# python main.py --prefix ada-exp3s-swd --data real-t --ada_algo exp3s --ada_discount window --ada_lr 3 --ada_epsilon 0.1 --ada_decay 0.9 --sampling adahybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/exp3s-wd-$run.out & 

# sleep 10
# python main.py --prefix ada-exp3-sdc  --data real-t --ada_algo exp3  --ada_discount decay  --ada_lr 3 --ada_epsilon 0.1 --ada_decay 0.9 --sampling adahybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/exp3-dc-$run.out & 

# sleep 10
# python main.py --prefix ada-exp3-swd  --data real-t --ada_algo exp3  --ada_discount window --ada_lr 3 --ada_epsilon 0.1 --ada_decay 0.9 --sampling adahybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/exp3-wd-$run.out & 

# sleep 10
# export CUDA_VISIBLE_DEVICES=1 && python main.py --prefix rada-mul  --drift pot --mixing multiply --data real-t --ada_algo exp3  --ada_discount window --ada_lr 3 --ada_epsilon 0.1 --ada_decay 0.9 --sampling rada --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/rada-mul-$run.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=0 && python main.py --prefix rada-bal-s  --drift pot --mixing balance --data real-t --ada_algo exp3s  --ada_discount decay --ada_lr 3 --ada_epsilon 0.1 --ada_decay 0.9 --sampling rada --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/rada-balance-$run.out & 

done