import argparse
import logging
import os
import random
import glob
import json
import math
import re

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from tensorboardX import SummaryWriter

from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer,
                          RobertaConfig, RobertaTokenizer)
from transformers import (XLMRobertaTokenizer, XLMRobertaConfig)
from transformers import (AdamW, get_linear_schedule_with_warmup)

# from modeling_bert_domain_classifier_multiple import (BertForDomainClassifier)
from modeling_xlmr_domain_classifier_multiple import (RobertaForDomainClassifier)
from data_processors import PROCESSORS
from utils_dst_domain_classifier_multiple import (convert_examples_to_features)

from tensorlistdataset import (TensorListDataset)

logger = logging.getLogger(__name__)

ALL_MODELS = tuple(BertConfig.pretrained_config_archive_map.keys())
ALL_MODELS += tuple(XLMRobertaConfig.pretrained_config_archive_map.keys())

MODEL_CLASSES = {
    # 'bert': (BertConfig, BertForDomainClassifier, BertTokenizer),
    'roberta': (XLMRobertaConfig, RobertaForDomainClassifier, XLMRobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def batch_to_device(batch, device):
    batch_on_device = []
    for element in batch:
        if isinstance(element, dict):
            batch_on_device.append({k: v.to(device) for k, v in element.items()})
        else:
            batch_on_device.append(element.to(device))
    return tuple(batch_on_device)


def train(args, train_dataset, features, model, tokenizer, processor, continue_from_global_step=0):
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

    if args.save_epochs > 0:
        args.save_steps = t_total // args.num_train_epochs * args.save_epochs

    num_warmup_steps = int(t_total * args.warmup_proportion)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    model_single_gpu = model
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model_single_gpu)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup steps = %d", num_warmup_steps)

    if continue_from_global_step > 0:
        logger.info("Fast forwarding to global step %d to resume training from latest checkpoint...",
                    continue_from_global_step)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            # If training is continued from a checkpoint, fast forward
            # to the state of that checkpoint.
            if global_step < continue_from_global_step:
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scheduler.step()  # Update learning rate schedule
                    global_step += 1
                continue

            model.train()
            batch = batch_to_device(batch, args.device)

            # This is what is forwarded to the "forward" def.
            inputs = {'input_ids': batch[0],
                      'input_mask': batch[1],
                      'segment_ids': batch[2],
                      'class_label_id': batch[8]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                epoch_iterator.update(1)
                epoch_iterator.set_postfix_str(f'Train Loss: {tr_loss / global_step}')

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss / global_step)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
            results = evaluate(args, model_single_gpu, tokenizer, processor, prefix=global_step)
            for key, value in results.items():
                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, processor, prefix=""):
    dataset, features = load_and_cache_examples(args, model, tokenizer, processor, evaluate=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(dataset)  # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    all_preds = []
    ds = {slot: 'none' for slot in model.slot_list}
    with torch.no_grad():
        diag_state = {slot: torch.tensor([0 for _ in range(args.eval_batch_size)]).to(args.device) for slot in
                      model.slot_list}
    count = 0
    count_max = len(model.slot_list)

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = batch_to_device(batch, args.device)

        with torch.no_grad():

            inputs = {'input_ids': batch[0],
                      'input_mask': batch[1],
                      'segment_ids': batch[2],
                      'class_label_id': batch[8]}
            unique_ids = [features[i.item()].guid for i in batch[9]]
            input_ids_unmasked = [features[i.item()].input_ids_unmasked for i in batch[9]]

            outputs = model(**inputs)

            outputs_total_loss = outputs[0]
            outputs_class_logits = outputs[2]
            unique_ids = unique_ids
            input_ids_unmasked = input_ids_unmasked

        results = eval_metric(model, inputs, outputs_total_loss, outputs_class_logits, args, tokenizer)
        all_results.append(results)
        count = 0

    # Generate final results1-positive-examples
    final_results = {}
    for k in all_results[0].keys():
        final_results[k] = torch.stack([r[k] for r in all_results]).mean()

    return final_results


def eval_metric(model, features, total_loss, per_slot_class_logits, args, tokenizer):
    metric_dict = {}
    per_slot_correctness = {}
    class_logits = per_slot_class_logits

    class_label_id = features['class_label_id']
    domain_list = ['taxi', 'restaurant', 'hotel', 'attraction', 'train']
    all_class_prediction = []
    for domain in domain_list:
        # logger.info('class logits %s', class_logits[domain])
        class_prob = torch.softmax(class_logits[domain], dim=-1)[:, 1]
        if class_prob >= 0.05:
            class_prediction = 1.0
        else:
            class_prediction = 0.0
        # _, class_prediction = class_logits[domain].max(1)
        all_class_prediction.append(class_prediction)
    class_prediction = torch.tensor(all_class_prediction).to(args.device)
    class_correctness = torch.equal(class_prediction, class_label_id.view(-1))
    if class_correctness:
        class_accuracy = torch.tensor(1).float()
        # if len(torch.where(class_prediction == 1)[0]) >= 2:
        #     logger.info('prediction %s', class_prediction)
    else:
        class_accuracy = torch.tensor(0).float()
        # logger.info('input ids %s', features['input_ids'][0])
        input_tokens = tokenizer.convert_ids_to_tokens(features['input_ids'][0])
        input_tokens = (' '.join(input_tokens)).replace(' ', '')
        input_tokens = re.sub("▁", " ", input_tokens)
        input_tokens = input_tokens.lower()
        # logger.info('sentence %s', input_tokens)
        # logger.info('prediction %s', class_prediction)
        # logger.info('ground truth %s', class_label_id.view(-1))
    total_correctness = class_correctness
    total_accuracy = total_correctness

    loss = total_loss
    metric_dict['eval_accuracy_class'] = class_accuracy
    metric_dict['loss'] = total_loss
    return metric_dict


def predict_and_format(model, tokenizer, features, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits,
                       ids, input_ids_unmasked, values, inform, prefix, ds, collect_domain_flag, args):
    prediction_list = []
    dialog_state = ds
    for i in range(len(ids)):
        if int(ids[i].split("-")[2]) == 0:
            dialog_state = {slot: 'none' for slot in model.slot_list}

        prediction = {}
        prediction_addendum = {}
        for slot in model.slot_list:
            class_logits = per_slot_class_logits[slot][i]
            start_logits = per_slot_start_logits[slot][i]
            end_logits = per_slot_end_logits[slot][i]

            input_ids = features[slot]['input_ids'][i].tolist()
            class_label_id = int(features[slot]['class_label_id'][i])
            start_pos = int(features[slot]['start_pos'][i])
            end_pos = int(features[slot]['end_pos'][i])

            if collect_domain_flag[slot]:
                class_prediction = int(class_logits.argmax())
                start_prediction = int(start_logits.argmax())
                end_prediction = int(end_logits.argmax())
            else:
                class_prediction = int(0)
                start_prediction = int(0)
                end_prediction = int(0)
            prediction['guid'] = ids[i].split("-")
            prediction['class_prediction_%s' % slot] = class_prediction
            prediction['class_label_id_%s' % slot] = class_label_id
            prediction['start_prediction_%s' % slot] = start_prediction
            prediction['start_pos_%s' % slot] = start_pos
            prediction['end_prediction_%s' % slot] = end_prediction
            prediction['end_pos_%s' % slot] = end_pos
            prediction['input_ids_%s' % slot] = input_ids

            if class_prediction == model.class_types.index('dontcare'):
                dialog_state[slot] = 'dontcare'
            elif class_prediction == model.class_types.index('copy_value'):
                input_tokens = tokenizer.convert_ids_to_tokens(input_ids_unmasked[slot][i])
                if args.model_type == 'roberta':
                    dialog_state[slot] = (' '.join(input_tokens[start_prediction:end_prediction + 1])).replace(' ', '')
                    dialog_state[slot] = re.sub("▁", " ", dialog_state[slot])
                    dialog_state[slot] = dialog_state[slot].lower()
                else:
                    dialog_state[slot] = ' '.join(input_tokens[start_prediction:end_prediction + 1])
                    dialog_state[slot] = re.sub("(^| )##", "", dialog_state[slot])
            elif 'true' in model.class_types and class_prediction == model.class_types.index('true'):
                dialog_state[slot] = 'true'
            elif 'false' in model.class_types and class_prediction == model.class_types.index('false'):
                dialog_state[slot] = 'false'
            elif class_prediction == model.class_types.index('inform'):
                dialog_state[slot] = '§§' + inform[slot][i]
            # Referral case is handled below
            prediction_addendum['slot_prediction_%s' % slot] = dialog_state[slot]
            prediction_addendum['slot_groundtruth_%s' % slot] = values[slot][i]

        prediction.update(prediction_addendum)
        prediction_list.append(prediction)

    return prediction_list, dialog_state


def load_and_cache_examples(args, model, tokenizer, processor, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_file = os.path.join(os.path.dirname(args.output_dir), 'cached_{}_features'.format(
        args.predict_type if evaluate else 'train'))
    if os.path.exists(cached_file) and not args.overwrite_cache:  # and not output_examples:
        logger.info("Loading features from cached file %s", cached_file)
        features = torch.load(cached_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        processor_args = {'append_history': args.append_history,
                          'use_history_labels': args.use_history_labels,
                          'swap_utterances': args.swap_utterances,
                          'label_value_repetitions': args.label_value_repetitions,
                          'delexicalize_sys_utts': args.delexicalize_sys_utts,
                          'unk_token': '<unk>' if args.model_type == 'roberta' else '[UNK]'}
        if evaluate and args.predict_type == "dev":
            examples = processor.get_dev_examples(args.data_dir, processor_args)
        elif evaluate and args.predict_type == "test":
            examples = processor.get_test_examples(args.data_dir, processor_args)
        else:
            examples = processor.get_train_examples(args.data_dir, processor_args)
        features = convert_examples_to_features(examples=examples,
                                                slot_list=model.slot_list,
                                                class_types=model.class_types,
                                                model_type=args.model_type,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                slot_value_dropout=(0.0 if evaluate else args.svd),
                                                evaluate=evaluate)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_file)
            torch.save(features, cached_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    all_start_positions = torch.tensor([f.start_pos for f in features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_pos for f in features], dtype=torch.long)
    all_inform_slot_ids = torch.tensor([f.inform_slot for f in features], dtype=torch.long)
    all_refer_ids = torch.tensor([f.refer_id for f in features], dtype=torch.long)
    all_diag_state = torch.tensor([f.diag_state for f in features], dtype=torch.long)
    all_class_label_ids = torch.tensor([f.class_label_id for f in features], dtype=torch.long)
    dataset = TensorListDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_inform_slot_ids,
                                all_refer_ids,
                                all_diag_state,
                                all_class_label_ids, all_example_index)

    return dataset, features


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="Name of the task (e.g., multiwoz21).")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="Task database.")
    parser.add_argument("--dataset_config", default=None, type=str, required=True,
                        help="Dataset configuration file.")
    parser.add_argument("--predict_type", default=None, type=str, required=True,
                        help="Portion of the data to perform prediction on (e.g., dev, test).")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="Maximum input length after tokenization. Longer sequences will be truncated, shorter ones padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the <predict_type> set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--dropout_rate", default=0.3, type=float,
                        help="Dropout rate for BERT representations.")
    parser.add_argument("--heads_dropout", default=0.0, type=float,
                        help="Dropout rate for classification heads.")
    parser.add_argument("--class_loss_ratio", default=0.8, type=float,
                        help="The ratio applied on class loss in total loss calculation. "
                             "Should be a value in [0.0, 1.0]. "
                             "The ratio applied on token loss is (1-class_loss_ratio)/2. "
                             "The ratio applied on refer loss is (1-class_loss_ratio)/2.")
    parser.add_argument("--token_loss_for_nonpointable", action='store_true',
                        help="Whether the token loss for classes other than copy_value contribute towards total loss.")
    parser.add_argument("--refer_loss_for_nonpointable", action='store_true',
                        help="Whether the refer loss for classes other than refer contribute towards total loss.")

    parser.add_argument("--append_history", action='store_true',
                        help="Whether or not to append the dialog history to each turn.")
    parser.add_argument("--use_history_labels", action='store_true',
                        help="Whether or not to label the history as well.")
    parser.add_argument("--swap_utterances", action='store_true',
                        help="Whether or not to swap the turn utterances (default: sys|usr, swapped: usr|sys).")
    parser.add_argument("--label_value_repetitions", action='store_true',
                        help="Whether or not to label values that have been mentioned before.")
    parser.add_argument("--delexicalize_sys_utts", action='store_true',
                        help="Whether or not to delexicalize the system utterances.")
    parser.add_argument("--class_aux_feats_inform", action='store_true',
                        help="Whether or not to use the identity of informed slots as auxiliary featurs for class prediction.")
    parser.add_argument("--class_aux_feats_ds", action='store_true',
                        help="Whether or not to use the identity of slots in the current dialog state as auxiliary featurs for class prediction.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.0, type=float,
                        help="Linear warmup over warmup_proportion * steps.")
    parser.add_argument("--svd", default=0.0, type=float,
                        help="Slot value dropout ratio (default: 0.0)")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=0,
                        help="Save checkpoint every X updates steps. Overwritten by --save_epochs.")
    parser.add_argument('--save_epochs', type=int, default=0,
                        help="Save checkpoint every X epochs. Overrides --save_steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    args = parser.parse_args()

    assert (args.warmup_proportion >= 0.0 and args.warmup_proportion <= 1.0)
    assert (args.svd >= 0.0 and args.svd <= 1.0)

    task_name = args.task_name.lower()
    if task_name not in PROCESSORS:
        raise ValueError("Task not found: %s" % (task_name))

    processor = PROCESSORS[task_name](args.dataset_config)
    dst_slot_list = processor.slot_list
    dst_class_types = processor.class_types
    dst_class_labels = len(dst_class_types)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)

    # Add DST specific parameters to config
    config.dst_dropout_rate = args.dropout_rate
    config.dst_heads_dropout_rate = args.heads_dropout
    config.dst_class_loss_ratio = args.class_loss_ratio
    config.dst_token_loss_for_nonpointable = args.token_loss_for_nonpointable
    config.dst_refer_loss_for_nonpointable = args.refer_loss_for_nonpointable
    config.dst_class_aux_feats_inform = args.class_aux_feats_inform
    config.dst_class_aux_feats_ds = args.class_aux_feats_ds
    config.dst_slot_list = dst_slot_list
    config.dst_class_types = dst_class_types
    config.dst_class_labels = dst_class_labels

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)

    # add special tokens <domain>, </domain>
    special_tokens_dict = {'additional_special_tokens': ['<domain>', '</domain>', '<slot>', '</slot>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)
    model.resize_token_embeddings(len(tokenizer))

    logger.info("Updated model config: %s" % config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        # If output files already exists, assume to continue training from latest checkpoint (unless overwrite_output_dir is set)
        continue_from_global_step = 0  # If set to 0, start training from the beginning
        if os.path.exists(args.output_dir) and os.listdir(
                args.output_dir) and args.do_train and not args.overwrite_output_dir:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/*/' + WEIGHTS_NAME, recursive=True)))
            if len(checkpoints) > 0:
                checkpoint = checkpoints[-1]
                logger.info("Resuming training from the latest checkpoint: %s", checkpoint)
                continue_from_global_step = int(checkpoint.split('-')[-1])
                model = model_class.from_pretrained(checkpoint)
                model.to(args.device)

        train_dataset, features = load_and_cache_examples(args, model, tokenizer, processor, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, features, model, tokenizer, processor,
                                     continue_from_global_step)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = []
    if args.do_eval and args.local_rank in [-1, 0]:
        output_eval_file = os.path.join(args.output_dir, "eval_res.%s.json" % (args.predict_type))
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for cItr, checkpoint in enumerate(checkpoints):
            # Reload the model
            global_step = checkpoint.split('-')[-1]
            if cItr == len(checkpoints) - 1:
                global_step = "final"
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, processor, prefix=global_step)
            result_dict = {k: float(v) for k, v in result.items()}
            result_dict["global_step"] = global_step
            results.append(result_dict)

            for key in sorted(result_dict.keys()):
                logger.info("%s = %s", key, str(result_dict[key]))

        with open(output_eval_file, "w") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    main()
