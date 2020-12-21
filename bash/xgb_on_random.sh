sleep 5
export CUDA_VISIBLE_DEVICES=0 && nohup python main.py --data real-t --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-t1-xgb-random-10.out & 

sleep 5
export CUDA_VISIBLE_DEVICES=1 && nohup python main.py --data real-n --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-n1-xgb-random-10.out &

sleep 5
export CUDA_VISIBLE_DEVICES=2 && nohup python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m1-xgb-random-10.out &

sleep 5
export CUDA_VISIBLE_DEVICES=3 && python main.py --data real-t --semi_supervised 0 --sampling xgb --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-t1-xgb-10.out & 

sleep 5
export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 0 --sampling xgb --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-n1-xgb-10.out & 

sleep 5
export CUDA_VISIBLE_DEVICES=1 && python main.py --data real-m --semi_supervised 0 --sampling xgb --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m1-xgb-10.out & 


sleep 5
export CUDA_VISIBLE_DEVICES=2 && nohup python main.py --data real-t --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-t2-xgb-random-10.out & 

sleep 5
export CUDA_VISIBLE_DEVICES=3 && nohup python main.py --data real-n --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-n2-xgb-random-10.out &

sleep 5
export CUDA_VISIBLE_DEVICES=0 && nohup python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m1-xgb-random-10.out &

sleep 5
export CUDA_VISIBLE_DEVICES=1 && python main.py --data real-t --semi_supervised 0 --sampling xgb --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-t1-xgb-10.out & 

sleep 5
export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-n --semi_supervised 0 --sampling xgb --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-n1-xgb-10.out & 

sleep 5
export CUDA_VISIBLE_DEVICES=3 && python main.py --data real-m --semi_supervised 0 --sampling xgb --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m1-xgb-10.out & 