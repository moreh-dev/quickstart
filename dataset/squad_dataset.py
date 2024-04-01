import collections
import json
from pathlib import Path

import numpy as np
import torch
from evaluate import load
from loguru import logger


class SquadDataset(torch.utils.data.Dataset):

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items() if key not in ['offset_mapping', 'example_id']
        }

    def __len__(self):
        return len(self.encodings['input_ids'])


def convert_to_features(args, contexts, answers, tokenizer):

    input_encodings = tokenizer.batch_encode_plus(contexts,
                                                  max_length=args.max_input_length,
                                                  truncation=True,
                                                  pad_to_max_length=True)

    target_encodings = tokenizer.batch_encode_plus(answers,
                                                   max_length=args.max_target_length,
                                                   truncation=True,
                                                   pad_to_max_length=True)

    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'decoder_input_ids': target_encodings['input_ids'],
        'decoder_attention_mask': target_encodings['attention_mask']
    }

    return encodings


def seq2seq_preprocess(questions, contexts, answers):

    adder = lambda x, y: 'question: ' + x + '  context: ' + y
    qcs = list(map(adder, questions, contexts))

    return qcs, answers


def collate_batch(batch):
    """
    Take a list of samples from a Dataset and collate them into a batch.
    Return:
        A dictionary of tensors
    """
    input_ids = torch.stack([example['input_ids'] for example in batch])
    lm_labels = torch.stack([example['decoder_input_ids'] for example in batch])
    attention_mask = torch.stack([example['attention_mask'] for example in batch])
    decoder_attention_mask = torch.stack([example['decoder_attention_mask'] for example in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'lm_labels': lm_labels,
        'decoder_attention_mask': decoder_attention_mask
    }


def compute_metrics(args, start_logits, end_logits, references, ref_ids, ref_context, val_ids, offset):
    metric = load("squad_v2")

    null_score_diff_threshold = 0.0

    all_predictions = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    example_id_to_index = {k: i for i, k in enumerate(ref_ids)}
    features_per_example = collections.defaultdict(list)

    for idx, feature in enumerate(val_ids):
        features_per_example[example_id_to_index[feature]].append(idx)

    cnt = 0
    n_best_size = 20
    for example_index, example in enumerate(ref_context):

        feature_indices = features_per_example[example_index]
        context = example
        prelim_predictions = []
        min_null_prediction = None

        # Loop through all features associated with that example
        for feature_index in feature_indices:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offset_mapping = offset[feature_index]

            feature_null_score = start_logit[0] + end_logit[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            token_is_max_context = None

            start_indexes = np.argsort(start_logit)[-1:-n_best_size - 1:-1].tolist()
            end_indexes = np.argsort(end_logit)[-1:-n_best_size - 1:-1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if (start_index >= len(offset_mapping) or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None or len(offset_mapping[start_index]) < 2
                            or offset_mapping[end_index] is None or len(offset_mapping[end_index]) < 2):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > args.max_target_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    #if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                    #    continue

                    prelim_predictions.append({
                        "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                        "score": start_logit[start_index] + end_logit[end_index],
                        "start_logit": start_logit[start_index],
                        "end_logit": end_logit[end_index],
                    })
        if min_null_prediction is not None:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]
        # Select the answer with the best score

        if (min_null_prediction is not None and not any(p["offsets"] == (0, 0) for p in predictions)):
            predictions.append(min_null_prediction)

        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0]:offsets[1]]

        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob
        j = 0
        while predictions[j]["text"] == "":
            j += 1
        best_non_null_pred = predictions[j]

        # Then we compare to the null prediction using the threshold.
        score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]

        if score_diff > null_score_diff_threshold:
            all_predictions[example_index] = ""
            cnt += 1
        else:
            all_predictions[example_index] = best_non_null_pred["text"]

    formatted_predictions = [{
        "id": str(k),
        "prediction_text": v,
        "no_answer_probability": 0.0
    } for k, (key, v) in enumerate(all_predictions.items())]
    theoretical_answers = [{
        "id": str(k),
        'answers': {
            'answer_start': [0],
            'text': [] if ans == '[]' else ans
        }
    } for k, ans in enumerate(references)]
    logger.info(f'{cnt} the number of pred lower than the thershold')

    return metric.compute(references=theoretical_answers, predictions=formatted_predictions)


def preprocess_training_examples(args, tokenizer, questions, contexts, answers):
    questions = [q.strip() for q in questions]
    inputs = tokenizer(
        questions,
        contexts,
        max_length=args.max_input_length,
        truncation="only_second",
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")

    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        try:
            start_char = answer["answer_start"]
            end_char = answer["answer_start"] + len(answer["text"])
            sequence_ids = inputs.sequence_ids(i)
        except:
            start_positions.append(0)
            end_positions.append(0)
            continue

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        try:
            while sequence_ids[idx] == 1:
                idx += 1
        except:
            idx = len(sequence_ids)
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs


def preprocess_validation_examples(args, tokenizer, questions, contexts, answers, idxs):
    questions = [q.strip() for q in questions]
    inputs = tokenizer(
        questions,
        contexts,
        max_length=args.max_input_length,
        truncation="only_second",
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(idxs[sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)]

    inputs["example_id"] = example_ids

    return inputs


def read_squad(path, seq2seq=True, valid=False):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

        contexts = []
        questions = []
        answers = []
        references = []
        ids = []
        for group in squad_dict['data']:
            for passage in group['paragraphs']:
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']
                    contexts.append(context)
                    questions.append(question)
                    if valid:
                        ids.append(qa['id'])

                    if seq2seq:
                        try:
                            answers.append(qa['answers'][0]['text'])
                        except:
                            answers.append('[]')
                    else:
                        try:
                            answers.append(qa['answers'][0])
                        except:
                            answers.append('')

                    if valid:
                        temp_list = []
                        for answer in qa['answers']:
                            temp_list.append(answer['text'])
                        if len(temp_list) == 0:
                            temp_list.append('[]')
                        references.append(temp_list)

    if valid:
        return contexts, questions, answers, references, ids
    else:
        return contexts, questions, answers


def get_squadv2_dataset(args, tokenizer):

    logger.info('loading dataset')

    if args.model in ['gpt2', 'gptj', 'bloom', 'roformer']:
        # roformer for QA cannot use generate method
        # use casual_lm_like method
        args.model_type = 'casual_lm'
    else:
        args.model_type = 'seq2seq'

    if args.model_type != 'seq2seq':

        train_contexts, train_questions, train_answers = read_squad(f'{args.dataset}/train-v2.0.json',
                                                                    seq2seq=False,
                                                                    valid=False)
        val_contexts, val_questions, val_answers, references, ids = read_squad(f'{args.dataset}/dev-v2.0.json',
                                                                               seq2seq=False,
                                                                               valid=True)

        train_encodings = preprocess_training_examples(args, tokenizer, train_questions, train_contexts, train_answers)
        val_encodings = preprocess_validation_examples(args, tokenizer, val_questions, val_contexts, val_answers, ids)

        train_dataset = SquadDataset(train_encodings)
        val_dataset = SquadDataset(val_encodings)

    else:

        train_contexts, train_questions, train_answers = read_squad(f'{args.dataset}/train-v2.0.json',
                                                                    seq2seq=True,
                                                                    valid=False)
        val_contexts, val_questions, val_answers, references, ids = read_squad(f'{args.dataset}/dev-v2.0.json',
                                                                               seq2seq=True,
                                                                               valid=True)

        train_qc, train_answers = seq2seq_preprocess(train_questions, train_contexts, train_answers)
        val_qc, val_answers = seq2seq_preprocess(val_questions, val_contexts, val_answers)

        train_encodings = convert_to_features(args, train_qc, train_answers, tokenizer)
        val_encodings = convert_to_features(args, val_qc, val_answers, tokenizer)

        train_dataset = SquadDataset(train_encodings)
        val_dataset = SquadDataset(val_encodings)

    return train_dataset, val_dataset, references, val_contexts, val_encodings, ids


if __name__ == '__main__':

    class dummyarg:

        def __init__(self):
            pass

    from transformers import AutoTokenizer

    args = dummyarg()
    args.cache = '../cache'
    args.max_target_length = 64
    args.max_input_length = 512
    args.max_answer_length = 16
    args.pad_to_max_length = None
    args.model = 'gpt2'
    args.dataset = '/lst2210/new_home4/dongseok/masterLM/data/'

    tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-560m', padding_side='right')

    A, train_dataset, val_dataset = get_squadv2_dataset(args, tokenizer)
