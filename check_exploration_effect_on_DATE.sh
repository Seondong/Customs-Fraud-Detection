sleep 3
export CUDA_VISIBLE_DEVICES=0 && nohup python main.py --data real-t --semi_supervised 0 --sampling hybrid --subsamplings DATE/random --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-t5-DATE-random-5.out & 

sleep 3
export CUDA_VISIBLE_DEVICES=1 && nohup python main.py --data real-t --semi_supervised 0 --sampling hybrid --subsamplings DATE/diversity --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-t5-DATE-diversity-5.out & 

sleep 3
export CUDA_VISIBLE_DEVICES=2 && nohup python main.py --data real-t --semi_supervised 0 --sampling hybrid --subsamplings DATE/badge --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-t5-DATE-badge-5.out &

sleep 3
export CUDA_VISIBLE_DEVICES=3 && nohup python main.py --data real-t --semi_supervised 0 --sampling hybrid --subsamplings DATE/bATE --weights 0.9/0.1 --mode scratch --train_from 20150101 --test_from 20150301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-t5-DATE-bATE-5.out &

sleep 3
export CUDA_VISIBLE_DEVICES=0 && nohup python main.py --data real-n --semi_supervised 0 --sampling hybrid --subsamplings DATE/random --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-n5-DATE-random-5.out &

sleep 3
export CUDA_VISIBLE_DEVICES=1 && nohup python main.py --data real-n --semi_supervised 0 --sampling hybrid --subsamplings DATE/diversity --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-n5-DATE-diversity-5.out &

sleep 3
export CUDA_VISIBLE_DEVICES=2 && nohup python main.py --data real-n --semi_supervised 0 --sampling hybrid --subsamplings DATE/badge --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-n5-DATE-badge-5.out &

sleep 3
export CUDA_VISIBLE_DEVICES=3 && nohup python main.py --data real-n --semi_supervised 0 --sampling hybrid --subsamplings DATE/bATE --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-n5-DATE-bATE-5.out &

sleep 3
export CUDA_VISIBLE_DEVICES=0 && nohup python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings DATE/random --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-m5-DATE-random-5.out &

sleep 3
export CUDA_VISIBLE_DEVICES=1 && nohup python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings DATE/diversity --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-m5-DATE-diversity-5.out &

sleep 3
export CUDA_VISIBLE_DEVICES=2 && nohup python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings DATE/badge --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-m5-DATE-badge-5.out &

sleep 3
export CUDA_VISIBLE_DEVICES=3 && nohup python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings DATE/bATE --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-m5-DATE-bATE-5.out &
