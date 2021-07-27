# sleep 10
# export CUDA_VISIBLE_DEVICES=0 && python main.py --data synthetic --sampling random --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-s1-random-10.out & 

# sleep 10
# export CUDA_VISIBLE_DEVICES=1 && python main.py --data synthetic --sampling badge --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-s1-badge-10.out & 

# sleep 10
# export CUDA_VISIBLE_DEVICES=2 && python main.py --data synthetic --sampling bATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-s1-bATE-10.out & 

# sleep 10
# export CUDA_VISIBLE_DEVICES=3 && python main.py --data synthetic --sampling gATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-s1-gATE-10.out & 


sleep 10
export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --sampling random --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m2-random-10.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --sampling random --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m2-random-10.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-m --sampling random --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m2-random-10.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=1 && python main.py --data real-m --sampling badge --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m2-badge-10.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-m --sampling bATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m2-bATE-10.out & 

sleep 10
export CUDA_VISIBLE_DEVICES=3 && python main.py --data real-m --sampling enhanced_bATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-m2-enhaced_bATE-10.out & 


# sleep 10
# export CUDA_VISIBLE_DEVICES=4 && python main.py --data real-n --sampling random --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-n2-random-10.out & 

# sleep 10
# export CUDA_VISIBLE_DEVICES=5 && python main.py --data real-n --sampling badge --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-n2-badge-10.out & 

# sleep 10
# export CUDA_VISIBLE_DEVICES=0 && python main.py --data real-n --sampling bATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-n2-bATE-10.out & 

# sleep 10
# export CUDA_VISIBLE_DEVICES=1 && python main.py --data real-n --sampling enhanced_bATE --mode scratch --train_from 20130101 --test_from 20130201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-n2-enhanced_bATE-10.out & 


# sleep 10
# export CUDA_VISIBLE_DEVICES=2 && python main.py --data real-t --sampling random --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-t2-random-10.out & 

# sleep 10
# export CUDA_VISIBLE_DEVICES=3 && python main.py --data real-t --sampling badge --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-t2-badge-10.out & 

# sleep 10
# export CUDA_VISIBLE_DEVICES=4 && python main.py --data real-t --sampling bATE --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-t2-bATE-10.out & 

# sleep 10
# export CUDA_VISIBLE_DEVICES=5 && python main.py --data real-t --sampling enhanced_bATE --mode scratch --train_from 20150101 --test_from 20150201 --test_length 7 --valid_length 28 --initial_inspection_rate 100 --final_inspection_rate 10 --epoch 10 --closs bce --rloss full --save 0 --numweeks 300 --inspection_plan fast_linear_decay --batch_size 512 &> ./logs/fld-t2-enhanced_bATE-10.out & 