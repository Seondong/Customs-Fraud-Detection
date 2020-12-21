
    echo blah
    
    sleep 4

    export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings bATE/DATE --weights 0.1/0.9 --mode scratch --train_from 20130101 --test_from 20130701 --test_length 14 --valid_length 14 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay

    export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-m --semi_supervised 0 --sampling bATE --mode scratch --train_from 20130101 --test_from 20130701 --test_length 14 --valid_length 14 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-m --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130701 --test_length 14 --valid_length 14 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
        
    export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-m --semi_supervised 0 --sampling random --mode scratch --train_from 20130101 --test_from 20130701 --test_length 14 --valid_length 14 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-m --semi_supervised 0 --sampling xgb --mode scratch --train_from 20130101 --test_from 20130701 --test_length 14 --valid_length 14 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-m --semi_supervised 0 --sampling badge --mode scratch --train_from 20130101 --test_from 20130701 --test_length 14 --valid_length 14 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-m --semi_supervised 0 --sampling noupDATE --mode scratch --train_from 20130101 --test_from 20130701 --test_length 14 --valid_length 14 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-m --semi_supervised 0 --sampling randomupDATE --mode scratch --train_from 20130101 --test_from 20130701 --test_length 14 --valid_length 14 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    

    export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-t --semi_supervised 0 --sampling bATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-t --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
        
    export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-t --semi_supervised 0 --sampling random --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-t --semi_supervised 0 --sampling xgb --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-t --semi_supervised 0 --sampling badge --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-t --semi_supervised 0 --sampling noupDATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-t --semi_supervised 0 --sampling randomupDATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    
    
    export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-n --semi_supervised 0 --sampling bATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-n --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
        
    export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-n --semi_supervised 0 --sampling random --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-n --semi_supervised 0 --sampling xgb --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-n --semi_supervised 0 --sampling badge --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-n --semi_supervised 0 --sampling noupDATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    
    export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-n --semi_supervised 0 --sampling randomupDATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay
    