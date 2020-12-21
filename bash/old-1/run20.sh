export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-t --semi_supervised 0 --sampling random --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-t --semi_supervised 0 --sampling xgb --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-t --semi_supervised 0 --sampling xgb_lr --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-t --semi_supervised 0 --sampling DATE --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-t --semi_supervised 0 --sampling diversity --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-t --semi_supervised 0 --sampling badge --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-t --semi_supervised 0 --sampling bATE --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-t --semi_supervised 0 --sampling hybrid --subsamplings DATE/bATE --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-t --semi_supervised 0 --sampling tabnet --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-t --semi_supervised 1 --sampling ssl_ae --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

sleep .5

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 0 --sampling random --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 0 --sampling xgb --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 0 --sampling xgb_lr --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 0 --sampling diversity --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 0 --sampling badge --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 0 --sampling bATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 0 --sampling hybrid --subsamplings DATE/bATE --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 0 --sampling tabnet --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 1 --sampling ssl_ae --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

sleep .5

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling random --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling xgb --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling xgb_lr --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling diversity --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling badge --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling bATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings DATE/bATE --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling tabnet --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 1 --sampling ssl_ae --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs focal --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

sleep .5

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-t --semi_supervised 0 --sampling random --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-t --semi_supervised 0 --sampling xgb --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-t --semi_supervised 0 --sampling xgb_lr --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-t --semi_supervised 0 --sampling DATE --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-t --semi_supervised 0 --sampling diversity --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-t --semi_supervised 0 --sampling badge --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-t --semi_supervised 0 --sampling bATE --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-t --semi_supervised 0 --sampling hybrid --subsamplings DATE/bATE --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-t --semi_supervised 0 --sampling tabnet --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-t --semi_supervised 1 --sampling ssl_ae --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

sleep .5

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 0 --sampling random --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 0 --sampling xgb --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 0 --sampling xgb_lr --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 0 --sampling diversity --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 0 --sampling badge --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 0 --sampling bATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 0 --sampling hybrid --subsamplings DATE/bATE --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 0 --sampling tabnet --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 1 --sampling ssl_ae --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

sleep .5

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling random --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling xgb --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling xgb_lr --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling diversity --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling badge --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling bATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings DATE/bATE --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling tabnet --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay

export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 1 --sampling ssl_ae --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss masked --save 1 --numweeks 100 --inspection_plan fast_linear_decay
