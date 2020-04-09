python3 -m torch.distributed.launch --nproc_per_node 4 \
run_ner.py --data_dir /home/yeongjoon/data/Naver_NER/ \
--model_type bert \
--labels /home/yeongjoon/data/Naver_NER/labels.txt \
--model_name_or_path bert-base-multilingual-cased \
--output_dir /home/yeongjoon/documents/transformers/examples/ner/naver_model \
--max_seq_length  256 \
--num_train_epochs 3 \
--per_gpu_train_batch_size 8 \
--save_steps 2188 \
--seed 1 \
--do_train ;

python3 run_ner.py \
--data_dir /home/yeongjoon/data/Naver_NER/ \
--model_type bert \
--labels /home/yeongjoon/data/Naver_NER/labels.txt \
--model_name_or_path bert-base-multilingual-cased \
--output_dir /home/yeongjoon/documents/transformers/examples/ner/naver_model \
--max_seq_length  512 \
--per_gpu_train_batch_size 8 \
--save_steps 2188 \
--seed 1 \
--do_eval \
--do_predict
