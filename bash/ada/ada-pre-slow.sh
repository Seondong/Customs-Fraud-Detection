

sleep 10
export CUDA_VISIBLE_DEVICES=0 && python main.py --prefix ada-prelim --data real-m --sampling hybrid --subsamplings DATE/bATE --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-m-DATE-bATE.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=1 && python main.py --prefix ada-prelim --data real-m --sampling adahybrid --subsamplings DATE/bATE --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/ada-m-DATE-bATE.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=2 && python main.py --prefix ada-prelim --data real-t --sampling hybrid --subsamplings DATE/bATE --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/hybrid-t-DATE-bATE.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=3 && python main.py --prefix ada-prelim --data real-t --sampling adahybrid --subsamplings DATE/bATE --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/ada-t-DATE-bATE.out & 
