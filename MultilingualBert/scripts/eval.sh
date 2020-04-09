python3 run_ner.py --data_dir /home/yeongjoon/data/GermEVAL2014_NER/ \
--model_type bert \
--labels /home/yeongjoon/data/GermEVAL2014_NER/labels.txt \
--model_name_or_path bert-base-multilingual-cased \
--output_dir germeval-distributed-model \
--max_seq_length  128 \
--num_train_epochs 3 \
--per_gpu_train_batch_size 8 \
--save_steps 750 \
--seed 1 \
--do_eval \
--eval_all_checkpoints \

python3 run_ner.py --data_dir /home/yeongjoon/data/GermEVAL2014_NER/ \
--model_type bert \
--labels /home/yeongjoon/data/GermEVAL2014_NER/labels.txt \
--model_name_or_path bert-base-multilingual-cased \
--output_dir germeval-distributed-model \
--max_seq_length  128 \
--num_train_epochs 3 \
--per_gpu_train_batch_size 8 \
--save_steps 750 \
--seed 1 \
--do_predict;
