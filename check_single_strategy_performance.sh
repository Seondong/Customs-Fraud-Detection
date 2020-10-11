    echo check_single_strategy_performance - M dataset
    
    sleep 2
    
    
    python main.py --data real-m --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130701 --test_length 14 --valid_length 14 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    python main.py --data real-m --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130701 --test_length 14 --valid_length 14 --initial_inspection_rate 100 --final_inspection_rate 4.5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay

    
    echo check_single_strategy_performance - N dataset
    
    sleep 2
    
    
    python main.py --data real-n --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    python main.py --data real-n --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 4.5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    
    echo check_single_strategy_performance - t dataset
    
    sleep 2
    
    
    python main.py --data real-t --semi_supervised 0 --sampling DATE --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    python main.py --data real-t --semi_supervised 0 --sampling DATE --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 4.5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay