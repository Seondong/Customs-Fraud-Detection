for run in 1 2 3 4 5;
do

sleep 10
python main.py --prefix main-all --data real-c --sampling hybrid --subsamplings xgb/random --weights 0.0/1.0 --mode scratch --train_from 20160101 --test_from 20160301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/hybrid-c-0.out & 

sleep 10
python main.py --prefix main-all --data real-c --sampling hybrid --subsamplings xgb/random --weights 0.1/0.9 --mode scratch --train_from 20160101 --test_from 20160301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/hybrid-c-10.out & 

sleep 10
python main.py --prefix main-all --data real-c --sampling hybrid --subsamplings xgb/random --weights 0.2/0.8 --mode scratch --train_from 20160101 --test_from 20160301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/hybrid-c-20.out & 

sleep 10
python main.py --prefix main-all --data real-c --sampling hybrid --subsamplings xgb/random --weights 0.3/0.7 --mode scratch --train_from 20160101 --test_from 20160301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/hybrid-c-30.out & 

sleep 10
python main.py --prefix main-all --data real-c --sampling hybrid --subsamplings xgb/random --weights 0.4/0.6 --mode scratch --train_from 20160101 --test_from 20160301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/hybrid-c-40.out & 

sleep 10
python main.py --prefix main-all --data real-c --sampling hybrid --subsamplings xgb/random --weights 0.5/0.5 --mode scratch --train_from 20160101 --test_from 20160301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/hybrid-c-50.out & 

sleep 10
python main.py --prefix main-all --data real-c --sampling hybrid --subsamplings xgb/random --weights 0.6/0.4 --mode scratch --train_from 20160101 --test_from 20160301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/hybrid-c-60.out & 

sleep 10
python main.py --prefix main-all --data real-c --sampling hybrid --subsamplings xgb/random --weights 0.7/0.3 --mode scratch --train_from 20160101 --test_from 20160301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/hybrid-c-70.out & 

sleep 10
python main.py --prefix main-all --data real-c --sampling hybrid --subsamplings xgb/random --weights 0.8/0.2 --mode scratch --train_from 20160101 --test_from 20160301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/hybrid-c-80.out & 

sleep 10
python main.py --prefix main-all --data real-c --sampling hybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20160101 --test_from 20160301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/hybrid-c-90.out & 

sleep 10
python main.py --prefix main-all --data real-c --sampling hybrid --subsamplings xgb/random --weights 1.0/0.0 --mode scratch --train_from 20160101 --test_from 20160301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/hybrid-c-100.out & 

sleep 10
python main.py --prefix ada-exp3s-sdc --data real-c --ada_algo exp3s --ada_discount decay  --ada_lr 3 --ada_epsilon 0.1 --ada_decay 0.9 --sampling adahybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20160101 --test_from 20160301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/n-exp3s-dc-$run.out & 

# sleep 10
# python main.py --prefix ada-exp3s-swd --data real-c --ada_algo exp3s --ada_discount window --ada_lr 3 --ada_epsilon 0.1 --ada_decay 0.9 --sampling adahybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20160101 --test_from 20160301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/exp3s-wd-$run.out & 

# sleep 10
# python main.py --prefix ada-exp3-sdc  --data real-c --ada_algo exp3  --ada_discount decay  --ada_lr 3 --ada_epsilon 0.1 --ada_decay 0.9 --sampling adahybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20160101 --test_from 20160301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/exp3-dc-$run.out & 

sleep 10
python main.py --prefix ada-exp3-swd  --data real-c --ada_algo exp3  --ada_discount window --ada_lr 3 --ada_epsilon 0.1 --ada_decay 0.9 --sampling adahybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20160101 --test_from 20160301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/c-exp3-wd-$run.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=2 && python main.py --prefix rada-balance  --drift pot --mixing balance --data real-c --ada_algo exp3s  --ada_discount decay --ada_lr 3 --ada_epsilon 0.1 --ada_decay 0.9 --sampling rada --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20160101 --test_from 20160301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/c-rada-balance-$run.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=1 && python main.py --prefix rada-balance  --drift pot --mixing balance --data real-c --ada_algo exp3  --ada_discount window --ada_lr 3 --ada_epsilon 0.1 --ada_decay 0.9 --sampling rada --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20160101 --test_from 20160301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/c-rada-balances-$run.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=3 && python main.py --prefix main-cd --data real-c  --sampling pot --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20160101 --test_from 20160301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/c-pot-$run.out & 

done