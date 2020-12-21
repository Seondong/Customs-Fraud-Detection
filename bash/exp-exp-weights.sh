sleep 20
export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weight 0.01/0.99 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m1-xgb-random-10-weights-1-99.out & 

sleep 20
export CUDA_VISIBLE_DEVICES=1 && python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weight 0.02/0.98 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m1-xgb-random-10-weights-2-98.out & 

sleep 20
export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weight 0.05/0.95 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m1-xgb-random-10-weights-5-95.out & 

sleep 20
export CUDA_VISIBLE_DEVICES=3 && python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weight 0.1/0.9 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m1-xgb-random-10-weights-10-90.out & 

sleep 20
export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weight 0.2/0.8 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m1-xgb-random-10-weights-20-80.out & 

sleep 20
export CUDA_VISIBLE_DEVICES=1 && python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weight 0.3/0.7 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m1-xgb-random-10-weights-30-70.out & 

sleep 20
export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weight 0.4/0.6 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m1-xgb-random-10-weights-40-60.out & 

sleep 20
export CUDA_VISIBLE_DEVICES=3 && python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weight 0.5/0.5 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m1-xgb-random-10-weights-50-50.out & 

sleep 20
export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weight 0.6/0.4 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m1-xgb-random-10-weights-60-40.out & 

sleep 20
export CUDA_VISIBLE_DEVICES=1 && python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weight 0.7/0.3 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m1-xgb-random-10-weights-70-30.out & 

sleep 20
export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weight 0.8/0.2 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m1-xgb-random-10-weights-80-20.out & 

sleep 20
export CUDA_VISIBLE_DEVICES=3 && python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weight 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m1-xgb-random-10-weights-90-10.out & 

sleep 20
export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weight 0.95/0.05 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m1-xgb-random-10-weights-95-5.out & 

sleep 20
export CUDA_VISIBLE_DEVICES=1 && python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weight 0.98/0.02 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m1-xgb-random-10-weights-98-2.out &

sleep 20
export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weight 0.99/0.01 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m1-xgb-random-10-weights-99-1.out &