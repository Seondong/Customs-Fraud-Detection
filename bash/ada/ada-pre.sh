

sleep 10
python main.py --prefix ada-prelim --data real-m --sampling hybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-m-xgb-random.out & 

sleep 10
python main.py --prefix ada-prelim --data real-m --sampling adahybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/ada-m-xgb-random.out & 

sleep 10
python main.py --prefix ada-prelim --data real-t --sampling hybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-t-xgb-random.out & 

sleep 10
python main.py --prefix ada-prelim --data real-t --sampling adahybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/ada-t-xgb-random.out & 
