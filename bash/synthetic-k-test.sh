sleep 10
export CUDA_VISIBLE_DEVICES=0 && python main.py --data synthetic-k --batch_size 128 --sampling xgb --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 10 --final_inspection_rate 10 --epoch 5 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural &> ./logs/debugging-synk-xgb-10.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=1 && python main.py --data synthetic-k --batch_size 128 --sampling DATE --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 10 --final_inspection_rate 10 --epoch 5 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural &> ./logs/debugging-synk-DATE-10.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=2 && python main.py --data synthetic-k --batch_size 128 --sampling bATE --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 10 --final_inspection_rate 10 --epoch 5 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural &> ./logs/debugging-synk-bATE-10.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=2 && python main.py --data synthetic-k --batch_size 128 --sampling random --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 10 --final_inspection_rate 10 --epoch 5 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural &> ./logs/debugging-synk-random-10.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=0 && python main.py --data synthetic-k --batch_size 128 --sampling hybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 10 --final_inspection_rate 10 --epoch 5 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural &> ./logs/debugging-synk-hybrid-xgb+random-10.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=4 && python main.py --data synthetic-k --batch_size 128 --sampling hybrid --subsamplings DATE/random --weights 0.9/0.1 --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 10 --final_inspection_rate 10 --epoch 5 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural &> ./logs/debugging-synk-hybrid-DATE+random-10.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=3 && python main.py --data synthetic-k --batch_size 128 --sampling adahybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 10 --final_inspection_rate 10 --epoch 5 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural &> ./logs/debugging-synk-adahybrid-xgb+random-10.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=0 && python main.py --data synthetic-k --batch_size 128 --sampling pot --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 10 --final_inspection_rate 10 --epoch 5 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural &> ./logs/debugging-synk-pot-xgb+random-10.out & 
