sleep 10
export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-t --semi_supervised 0 --sampling xgb --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-t1-xgb-10.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=1 && python main.py --data real-t --semi_supervised 0 --sampling xgb --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-t2-xgb-10.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-t --semi_supervised 0 --sampling DATE --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-t1-DATE-10.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-t --semi_supervised 0 --sampling DATE --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-t2-DATE-10.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=3 && python main.py --data real-t --semi_supervised 1 --sampling ssl_ae --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-t1-ssl_ae-10.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=4 && python main.py --data real-t --semi_supervised 1 --sampling ssl_ae --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-t2-ssl_ae-10.out & 