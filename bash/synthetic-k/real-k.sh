python main.py --data real-k --prefix kcstest --batch_size 128 --sampling random --mode scratch --train_from 20200101 --test_from 20200131 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural;

python main.py --data real-k --prefix kcstest --batch_size 128 --sampling risky --mode scratch --train_from 20200101 --test_from 20200131 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural --risk_profile ratio;

python main.py --data real-k --prefix kcstest --batch_size 128 --sampling xgb --mode scratch --train_from 20200101 --test_from 20200131 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural;

python main.py --data real-k --prefix kcstest --batch_size 128 --sampling DATE --mode scratch --train_from 20200101 --test_from 20200131 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural;

python main.py --data real-k --prefix kcstest --batch_size 128 --sampling hybrid --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20200101 --test_from 20200131 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural --risk_profile ratio;

python main.py --data real-k --prefix kcstest --batch_size 128 --sampling hybrid --subsamplings DATE/bATE --weights 0.9/0.1 --mode scratch --train_from 20200101 --test_from 20200131 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural --risk_profile ratio;

python main.py --data real-k --prefix kcstest --batch_size 128 --sampling adahybrid --subsamplings DATE/bATE --weights 0.9/0.1 --mode scratch --train_from 20200101 --test_from 20200131 --test_length 7 --valid_length 7 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural --risk_profile ratio;