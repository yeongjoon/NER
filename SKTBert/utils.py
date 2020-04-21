import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from transformers import BertConfig, DistilBertConfig, BertTokenizer, BertForTokenClassification, DistilBertForTokenClassification
from tokenization_kobert import KoBertTokenizer

from modeling_NER_bert_crf import BertCRFForTokenClassification

MODEL_CLASSES = {
    'kobert': (BertConfig, BertForTokenClassification, KoBertTokenizer),
    'distilkobert': (DistilBertConfig, DistilBertForTokenClassification, KoBertTokenizer),
    'bert': (BertConfig, BertForTokenClassification, BertTokenizer),
    'kobert-lm': (BertConfig, BertForTokenClassification, KoBertTokenizer),
    'kobert-crf':(BertConfig, BertCRFForTokenClassification, KoBertTokenizer),
}

MODEL_PATH_MAP = {
    'kobert': 'monologg/kobert',
    'distilkobert': 'monologg/distilkobert',
    'multilingual-bert': 'bert-base-multilingual-cased',
    'kobert-lm': 'monologg/kobert-lm',
    'kobert-crf': 'monologg/kobert',
}

def init_logger(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        filename=args.log_filename,
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
