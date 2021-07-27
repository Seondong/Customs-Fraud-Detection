


sleep 10
python main.py --prefix ada --data real-t --sampling hybrid --subsamplings xgb/random --weights 0.0/1.0 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-t-0.out & 

sleep 10
python main.py --prefix ada --data real-t --sampling hybrid --subsamplings xgb/random --weights 0.1/0.9 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-t-10.out & 

sleep 10
python main.py --prefix ada --data real-t --sampling hybrid --subsamplings xgb/random --weights 0.2/0.8 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-t-20.out & 

sleep 10
python main.py --prefix ada --data real-t --sampling hybrid --subsamplings xgb/random --weights 0.3/0.7 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-t-30.out & 

sleep 10
python main.py --prefix ada --data real-t --sampling hybrid --subsamplings xgb/random --weights 0.4/0.6 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-t-40.out & 

sleep 10
python main.py --prefix ada --data real-t --sampling hybrid --subsamplings xgb/random --weights 0.5/0.5 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-t-50.out & 

sleep 10
python main.py --prefix ada --data real-t --sampling hybrid --subsamplings xgb/random --weights 0.6/0.4 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-t-60.out & 

sleep 10
python main.py --prefix ada --data real-t --sampling hybrid --subsamplings xgb/random --weights 0.7/0.3 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-t-70.out & 

sleep 10
python main.py --prefix ada --data real-t --sampling hybrid --subsamplings xgb/random --weights 0.8/0.2 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-t-80.out & 

sleep 10
python main.py --prefix ada --data real-t --sampling hybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-t-90.out & 

sleep 10
python main.py --prefix ada --data real-t --sampling hybrid --subsamplings xgb/random --weights 1.0/0.0 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-t-100.out & 

sleep 10
python main.py --prefix ada --data real-t --sampling adahybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/ada-t.out & 
