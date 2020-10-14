sleep 5
export CUDA_VISIBLE_DEVICES=0 && nohup python main.py --data real-t --semi_supervised 0 --sampling hybrid --subsamplings DATE/random --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 256 &> ./logs/fld-t1-DATE-random-10.out & 

sleep 5
export CUDA_VISIBLE_DEVICES=1 && nohup python main.py --data real-t --semi_supervised 0 --sampling hybrid --subsamplings DATE/badge --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 256 &> ./logs/fld-t1-DATE-badge-10.out &

sleep 5
export CUDA_VISIBLE_DEVICES=2 && nohup python main.py --data real-t --semi_supervised 0 --sampling hybrid --subsamplings DATE/bATE --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 256 &> ./logs/fld-t1-DATE-bATE-10.out &

sleep 5
export CUDA_VISIBLE_DEVICES=3 && python main.py --data real-t --semi_supervised 0 --sampling DATE --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 256 &> ./logs/fld-t1-DATE-10.out & 




sleep 5
export CUDA_VISIBLE_DEVICES=3 && nohup python main.py --data real-n --semi_supervised 0 --sampling hybrid --subsamplings DATE/random --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 256 &> ./logs/fld-n1-DATE-random-10.out &

sleep 5
export CUDA_VISIBLE_DEVICES=2 && nohup python main.py --data real-n --semi_supervised 0 --sampling hybrid --subsamplings DATE/badge --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 256 &> ./logs/fld-n1-DATE-badge-10.out &

sleep 5
export CUDA_VISIBLE_DEVICES=1 && nohup python main.py --data real-n --semi_supervised 0 --sampling hybrid --subsamplings DATE/bATE --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 256 &> ./logs/fld-n1-DATE-bATE-10.out &

sleep 5
export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 256 &> ./logs/fld-n1-DATE-10.out & 



sleep 5
export CUDA_VISIBLE_DEVICES=0 && nohup python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings DATE/random --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 256 &> ./logs/fld-m1-DATE-random-10.out &

sleep 5
export CUDA_VISIBLE_DEVICES=1 && nohup python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings DATE/badge --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 256 &> ./logs/fld-m1-DATE-badge-10.out &

sleep 5
export CUDA_VISIBLE_DEVICES=2 && nohup python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings DATE/bATE --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 256 &> ./logs/fld-m1-DATE-bATE-10.out &

sleep 5
export CUDA_VISIBLE_DEVICES=3 && python main.py --data real-m --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 256 &> ./logs/fld-m1-DATE-10.out & 



sleep 5
export CUDA_VISIBLE_DEVICES=3 && nohup python main.py --data synthetic --semi_supervised 0 --sampling hybrid --subsamplings DATE/random --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 256 &> ./logs/fld-s1-DATE-random-10.out &

sleep 5
export CUDA_VISIBLE_DEVICES=2 && nohup python main.py --data synthetic --semi_supervised 0 --sampling hybrid --subsamplings DATE/badge --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 256 &> ./logs/fld-s1-DATE-badge-10.out &

sleep 5
export CUDA_VISIBLE_DEVICES=1 && nohup python main.py --data synthetic --semi_supervised 0 --sampling hybrid --subsamplings DATE/bATE --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 256 &> ./logs/fld-s1-DATE-bATE-10.out &

sleep 5
export CUDA_VISIBLE_DEVICES=0 && python main.py --data synthetic --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 256 &> ./logs/fld-s1-DATE-10.out & 




