

sleep 10
python main.py --prefix ada --data real-c --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weights 0.0/1.0 --mode scratch --train_from 20160101 --test_from 20160201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-c-0.out & 

sleep 10
python main.py --prefix ada --data real-c --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weights 0.1/0.9 --mode scratch --train_from 20160101 --test_from 20160201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-c-10.out & 

sleep 10
python main.py --prefix ada --data real-c --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weights 0.2/0.8 --mode scratch --train_from 20160101 --test_from 20160201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-c-20.out & 

sleep 10
python main.py --prefix ada --data real-c --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weights 0.3/0.7 --mode scratch --train_from 20160101 --test_from 20160201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-c-30.out & 

sleep 10
python main.py --prefix ada --data real-c --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weights 0.4/0.6 --mode scratch --train_from 20160101 --test_from 20160201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-c-40.out & 

sleep 10
python main.py --prefix ada --data real-c --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weights 0.5/0.5 --mode scratch --train_from 20160101 --test_from 20160201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-c-50.out & 

sleep 10
python main.py --prefix ada --data real-c --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weights 0.6/0.4 --mode scratch --train_from 20160101 --test_from 20160201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-c-60.out & 

sleep 10
python main.py --prefix ada --data real-c --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weights 0.7/0.3 --mode scratch --train_from 20160101 --test_from 20160201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-c-70.out & 

sleep 10
python main.py --prefix ada --data real-c --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weights 0.8/0.2 --mode scratch --train_from 20160101 --test_from 20160201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-c-80.out & 

sleep 10
python main.py --prefix ada --data real-c --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20160101 --test_from 20160201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-c-90.out & 

sleep 10
python main.py --prefix ada --data real-c --semi_supervised 0 --sampling hybrid --subsamplings xgb/random --weights 1.0/0.0 --mode scratch --train_from 20160101 --test_from 20160201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-c-100.out & 

sleep 10
python main.py --prefix ada --data real-c --semi_supervised 0 --sampling adahybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20160101 --test_from 20160201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/ada-c.out & 
