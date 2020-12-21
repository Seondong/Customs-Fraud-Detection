for i in {1..5}

do
    echo blah
    export CUDA_VISIBLE_DEVICES=3 && python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings bATE/DATE --weights 0.1/0.9 --mode scratch --train_from 20130101 --test_from 20130701 --test_length 30 --valid_length 30 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay
    
    export CUDA_VISIBLE_DEVICES=3 && python main.py --data real-t --semi_supervised 0 --sampling hybrid --subsamplings bATE/DATE --weights 0.1/0.9 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay
        
    export CUDA_VISIBLE_DEVICES=3 && python main.py --data real-n --semi_supervised 0 --sampling hybrid --subsamplings bATE/DATE --weights 0.1/0.9 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay


done