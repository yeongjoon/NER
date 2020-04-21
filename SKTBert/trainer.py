import os
import shutil
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from data_loader import load_and_cache_examples
from transformers import (
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from utils import set_seed
from data_loader import get_labels

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), ())

TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]

from tokenization_kobert import KoBertTokenizer

# CRF Adding

from modeling_NER_bert_crf import BertCRFForTokenClassification

class Trainer(object):
    def __init__(self, args):
        self.args = args
        # Setup logging
        if (
                os.path.exists(self.args.output_dir)
                and os.listdir(self.args.output_dir)
                and self.args.do_train
                and not self.args.overwrite_output_dir
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    self.args.output_dir
                )
            )

        # Setup distant debugging if needed
        if self.args.server_ip and self.args.server_port:
            # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
            import ptvsd

            print("Waiting for debugger attach")
            ptvsd.enable_attach(address=(self.args.server_ip, self.args.server_port), redirect_output=True)
            ptvsd.wait_for_attach()

        # Setup CUDA, GPU & distributed training
        if self.args.local_rank == -1 or self.args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
            self.args.n_gpu = 0 if self.args.no_cuda else torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.args.n_gpu = 1
        self.args.device = device

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            filename=self.args.log_filename,
            level=logging.INFO if self.args.local_rank in [-1, 0] else logging.WARN,
        )
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            self.args.local_rank,
            device,
            self.args.n_gpu,
            bool(self.args.local_rank != -1),
            self.args.fp16,
        )

        # Set seed
        set_seed(self.args)

        # Prepare CONLL-2003 task
        self.labels = get_labels(self.args.labels)
        self.num_labels = len(self.labels)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.results = {}

    def train(self):
        model, tokenizer = self._prepare_model(args=self.args, labels=self.labels, num_labels=self.num_labels, mode='train')
        train_dataset = load_and_cache_examples(args=self.args,
                                                tokenizer=tokenizer,
                                                labels=self.labels,
                                                pad_token_label_id=self.pad_token_label_id,
                                                mode="train")
        global_step, tr_loss = self._train(self.args, train_dataset, model, tokenizer,
                                           self.labels, self.pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    def evaluate(self):
        #TODO evaluate all checkpoint는 나중에 구현
        if self.args.local_rank in [-1, 0]:
            model, tokenizer = self._prepare_model(args=self.args, labels=self.labels, num_labels=self.num_labels, mode='eval')
            eval_dataset = load_and_cache_examples(args=self.args,
                                                   tokenizer=tokenizer,
                                                   labels=self.labels,
                                                   pad_token_label_id=self.pad_token_label_id,
                                                   mode='dev')
            result, _ = self._evaluate(self.args, model, eval_dataset, self.labels, self.pad_token_label_id,
                                       mode='dev', prefix="")
            self.results.update(result)
            output_eval_file = os.path.join(self.args.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(self.results.keys()):
                    writer.write("{} = {}\n".format(key, str(self.results[key])))

    def predict(self):
        if self.args.local_rank in [-1, 0]:
            model, tokenizer = self._prepare_model(args=self.args, labels=self.labels, num_labels=self.num_labels, mode='eval')
            test_dataset = load_and_cache_examples(args=self.args,
                                                   tokenizer=tokenizer,
                                                   labels=self.labels,
                                                   pad_token_label_id=self.pad_token_label_id,
                                                   mode='test')
            result, predictions = self._evaluate(self.args, model, test_dataset, self.labels, self.pad_token_label_id,
                                       mode='test', prefix="")

            output_test_results_file = os.path.join(self.args.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(result.keys()):
                    writer.write("{} = {}\n".format(key, str(result[key])))
            # Save predictions
            output_test_predictions_file = os.path.join(self.args.output_dir, "test_predictions.txt")
            with open(output_test_predictions_file, "w") as writer:
                with open(os.path.join(self.args.data_dir, "test.txt"), "r") as f:
                    example_id = 0
                    for line in f:
                        if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                            writer.write(line)
                            if not predictions[example_id]:
                                example_id += 1
                        elif predictions[example_id]:
                            output_line = line.split()[0] + " " + predictions[example_id].pop(0) + "\n"
                            writer.write(output_line)
                        else:
                            logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])

    def _train(self, args, train_dataset, model, tokenizer, labels, pad_token_label_id):
        """ Train the model """
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()

        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(args.model_name_or_path, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
            )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if os.path.exists(args.model_name_or_path):
            # set global_step to gobal_step of last saved checkpoint from model path
            try:
                global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
            except ValueError:
                global_step = 0
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
        )
        set_seed(args)  # Added here for reproductibility
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet"] else None
                    )  # XLM and RoBERTa don"t use segment_ids

                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        if (
                                args.local_rank == -1 and args.evaluate_during_training
                        ):  # Only evaluate when single GPU otherwise metrics may not average well
                            results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev")
                            for key, value in results.items():
                                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logging_loss = tr_loss

                    if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break
            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

        if args.local_rank in [-1, 0]:
            tb_writer.close()

        # Saves the last model when ended.
        self._save_model(args, model, tokenizer)

        return global_step, tr_loss / global_step

    def _evaluate(self, args, model, eval_dataset, labels, pad_token_label_id, mode, prefix=""):
        # eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu evaluate
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation %s *****", prefix)
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet"] else None
                    )  # XLM and RoBERTa don"t use segment_ids
                outputs = model(**inputs)
                # Predict와 Evalute 모두 호환될 수 있도록 조정
                tmp_eval_loss, preds_tensor = outputs[0], outputs[-1]

                if args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1
            if preds is None:
                preds = preds_tensor.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, preds_tensor.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        # preds = np.argmax(preds, axis=2) # 이미 CRF에서 decoding을 통해서 label을 출력하기 때문에 np.armax할 필요 없음

        label_map = {i: label for i, label in enumerate(labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        results = {
            "loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }

        logger.info("***** Eval results %s *****", prefix)
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results, preds_list

    def _prepare_model(self, args, labels, num_labels, mode='train'):
        """ prepare model and tokenizer for the trainer.
        :param args: parsed argument.
        :param labels: label list of NER.
        :param num_labels: number of labels.
        :return: pretrained model, tokenizer.
        """
        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        args.model_type = args.model_type.lower()
        config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            id2label={str(i): label for i, label in enumerate(labels)},
            label2id={label: i for i, label in enumerate(labels)},
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        tokenizer_args = {k: v for k, v in vars(args).items() if v is not None and k in TOKENIZER_ARGS}
        logger.info("Tokenizer arguments: %s", tokenizer_args)

        if mode == 'train':
            tokenizer = KoBertTokenizer.from_pretrained(
                args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                cache_dir=args.cache_dir if args.cache_dir else None,
                **tokenizer_args,
            )
        else:
            tokenizer = KoBertTokenizer.from_pretrained(args.output_dir, **tokenizer_args)
        # KoBERT tokenizer로 강제 지정

        # tokenizer = AutoTokenizer.from_pretrained(
        #     args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        #     cache_dir=args.cache_dir if args.cache_dir else None,
        #     **tokenizer_args,
        # )

        # CRF Adding
        if mode == 'train':
            model = BertCRFForTokenClassification.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
        else:
            logger.info("Evaluate the following checkpoints: %s", [args.output_dir])
            model = BertCRFForTokenClassification.from_pretrained(args.output_dir)

        # model = AutoModelForTokenClassification.from_pretrained(
        #     args.model_name_or_path,
        #     from_tf=bool(".ckpt" in args.model_name_or_path),
        #     config=config,
        #     cache_dir=args.cache_dir if args.cache_dir else None,
        # )

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(args.device)

        logger.info("Training/evaluation parameters %s", args)

        return model, tokenizer

    def _save_model(self, args, model, tokenizer):
        """
        Saves the last model when the training is ended.
        :param args: parsed argument.
        :param model: trained model(all epoch finished).
        :param tokenizer: corresponding tokenizer.
        :return:
        """
        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            # Create output directory if needed
            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(args.output_dir)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))