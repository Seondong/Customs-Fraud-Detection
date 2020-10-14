sleep 5
export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-m1-DATE-5.out & 

sleep 5
export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-n --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-n1-DATE-5.out & 

sleep 5
export CUDA_VISIBLE_DEVICES=3 && python main.py --data real-t --semi_supervised 0 --sampling DATE --mode scratch --train_from 20150101 --test_from 20150301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-t1-DATE-5.out & 

sleep 5
export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-m2-DATE-5.out & 

sleep 5
export CUDA_VISIBLE_DEVICES=1 && python main.py --data real-n --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-n2-DATE-5.out & 

sleep 5
export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-t --semi_supervised 0 --sampling DATE --mode scratch --train_from 20150101 --test_from 20150301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-t2-DATE-5.out & 

sleep 5
export CUDA_VISIBLE_DEVICES=3 && python main.py --data real-m --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-m3-DATE-5.out & 

sleep 5
export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-n3-DATE-5.out & 

sleep 5
export CUDA_VISIBLE_DEVICES=1 && python main.py --data real-t --semi_supervised 0 --sampling DATE --mode scratch --train_from 20150101 --test_from 20150301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-t3-DATE-5.out & 

sleep 5
export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-m --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-m4-DATE-5.out & 

sleep 5
export CUDA_VISIBLE_DEVICES=3 && python main.py --data real-n --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-n4-DATE-5.out & 

sleep 5
export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-t --semi_supervised 0 --sampling DATE --mode scratch --train_from 20150101 --test_from 20150301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-t4-DATE-5.out & 

sleep 5
export CUDA_VISIBLE_DEVICES=1 && python main.py --data real-m --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-m5-DATE-5.out & 

sleep 5
export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-n --semi_supervised 0 --sampling DATE --mode scratch --train_from 20130101 --test_from 20130301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-n5-DATE-5.out & 

sleep 5
export CUDA_VISIBLE_DEVICES=3 && python main.py --data real-t --semi_supervised 0 --sampling DATE --mode scratch --train_from 20150101 --test_from 20150301 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 5 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan direct_decay --batch_size 512 &> ./logs/nohup-t5-DATE-5.out & 
