for i in {1..10}
do
    sleep 5
    python main.py --data real-k --prefix zzzz --batch_size 128 --sampling risky --risk_profile ratio --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 100 --final_inspection_rate 20 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural &

    sleep 5
    python main.py --data real-k --prefix zzzz --batch_size 128 --sampling riskylogistic --risk_profile ratio --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 100 --final_inspection_rate 20 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural &

    sleep 5
    python main.py --data real-k --prefix zzzz --batch_size 128 --sampling riskyprod --risk_profile ratio --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 100 --final_inspection_rate 20 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural &

    sleep 5
    python main.py --data real-k --prefix zzzz --batch_size 128 --sampling riskyprec --risk_profile ratio --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 100 --final_inspection_rate 20 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural &

    sleep 5
    python main.py --data real-k --prefix zzzz --batch_size 128 --sampling riskyMAB --risk_profile ratio --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 100 --final_inspection_rate 20 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural &

    sleep 5
    python main.py --data real-k --prefix zzzz --batch_size 128 --sampling riskyMABsum --risk_profile ratio --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 100 --final_inspection_rate 20 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural &

    sleep 5
    python main.py --data real-k --prefix zzzz --batch_size 128 --sampling riskyDecayMAB --risk_profile ratio --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 100 --final_inspection_rate 20 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural &

    sleep 5
    python main.py --data real-k --prefix zzzz --batch_size 128 --sampling riskyDecayMABsum --risk_profile ratio --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 100 --final_inspection_rate 20 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural ;

    sleep 5
    python main.py --data real-k --prefix zzzz --batch_size 128 --sampling risky --risk_profile ratio --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 100 --final_inspection_rate 20 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural &

    sleep 5
    python main.py --data real-k --prefix zzzz --batch_size 128 --sampling riskylogistic --risk_profile ratio --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 100 --final_inspection_rate 20 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural &

    sleep 5
    python main.py --data real-k --prefix zzzz --batch_size 128 --sampling riskyprod --risk_profile ratio --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 100 --final_inspection_rate 20 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural &

    sleep 5
    python main.py --data real-k --prefix zzzz --batch_size 128 --sampling riskyprec --risk_profile ratio --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 100 --final_inspection_rate 20 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural &

    sleep 5
    python main.py --data real-k --prefix zzzz --batch_size 128 --sampling riskyMAB --risk_profile ratio --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 100 --final_inspection_rate 20 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural &

    sleep 5
    python main.py --data real-k --prefix zzzz --batch_size 128 --sampling riskyMABsum --risk_profile ratio --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 100 --final_inspection_rate 20 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural &

    sleep 5
    python main.py --data real-k --prefix zzzz --batch_size 128 --sampling riskyDecayMAB --risk_profile ratio --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 100 --final_inspection_rate 20 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural &

    sleep 5
    python main.py --data real-k --prefix zzzz --batch_size 128 --sampling riskyDecayMABsum --risk_profile ratio --mode scratch --train_from 20200101 --test_from 20200228 --test_length 15 --valid_length 15 --initial_inspection_rate 100 --final_inspection_rate 20 --epoch 2 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay --initial_masking natural ;
done