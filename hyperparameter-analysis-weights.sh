

    echo hyperparameter-analysis-weights - Nigeria dataset
    
    sleep 2
    
    
    python main.py --data real-n --semi_supervised 0 --sampling hybrid --subsamplings DATE/bATE --weights 0.6/0.4 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    python main.py --data real-n --semi_supervised 0 --sampling hybrid --subsamplings DATE/bATE --weights 0.8/0.2 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    python main.py --data real-n --semi_supervised 0 --sampling hybrid --subsamplings DATE/bATE --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    python main.py --data real-n --semi_supervised 0 --sampling hybrid --subsamplings DATE/bATE --weights 0.95/0.05 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    