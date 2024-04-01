import time

import torch
from loguru import logger

from hub_model.utils.batch_utils import BatchProcessor
from transformers.utils import ModelOutput

from hub_model.namespace import MetricNames
from hub_model.utils import (
    create_attn_mask,
    log_evaluation,
    logging_helper,
    mask_pads,
    metrics,
)
from hub_model.utils.metrics import maybe_evaluate_metric
from hub_model.utils.train_utils import ModelNames


def get_metrics(metric_names):
    return [getattr(metrics, getattr(MetricNames, x))() for x in metric_names]


# args can be hydra configuration
@torch.no_grad()
def evaluate(model, tokenizer, model_name, val_data_loader, args, epoch=0, metric_names: list = ['MeanLoss'], **kwargs):

    # TODO: check kwargs by model name
    model = model.eval()
    eval_metrics = get_metrics(metric_names)

    num_iterations = len(val_data_loader)
    log_this_step = logging_helper(num_iterations, args.eval_log_interval)
    log_interval_time = time.perf_counter()
    total_start = log_interval_time
    num_batches, num_tokens = 0, 0

    for (nbatch, batch) in enumerate(val_data_loader, 1):
        r"""
        inputs: Union[tokenized text inputs, images]
        labels: Union[language-model labels, image conditions]
        """
        processed_batch = BatchProcessor.get_processed_batch(model_name, args, batch, tokenizer)
        inputs, labels, attention_mask, batch_kwargs = processed_batch
        kwargs = {**kwargs, **batch_kwargs}
        num_losses = 1

        if model_name in [ModelNames.DIFFUSION]:
            loss = model(inputs.cuda())[0]
            outputs = ModelOutput(loss=loss, logits=None)

        elif model_name in [ModelNames.LLAMA2, ModelNames.QWEN, ModelNames.GPT, ModelNames.MISTRAL]:
            inputs, labels = (batch, mask_pads(batch, tokenizer))

            attention_mask = create_attn_mask(inputs, tokenizer).cuda()
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs, labels=labels, use_cache=False, attention_mask=attention_mask)

            if not isinstance(outputs, ModelOutput):
                outputs = ModelOutput(loss=outputs[0], logits=outputs[1])

        else:
            raise ValueError(
                f'{model_name} is not a valid model name. in case of adding new model, please add model case to evaluate function'
            )

        # hf style output format
        # loss, logit, etc
        num_batches += inputs.size(0)
        num_tokens += inputs.numel()
        lm_loss = outputs.loss
        lm_loss /= num_losses

        # add more maybe_evaluate_metric for additional custom metrics
        maybe_evaluate_metric(eval_metrics, 'loss', lm_loss)
        maybe_evaluate_metric(eval_metrics, 'input_and_label', (labels, outputs.logits, tokenizer))

        do_log, _ = log_this_step(nbatch)

        if do_log:
            # logger info logging tool
            log_interval_time = log_evaluation(
                args=args,
                epoch=epoch,
                log_interval_time=log_interval_time,
                iter_left=num_iterations - nbatch,
                iters=nbatch,
                loss=lm_loss,  # log current batch loss only
                num_batches=num_batches,
                processed_iters=nbatch,
                total_step=num_iterations,
            )

    total_duration = time.perf_counter() - total_start
    baselog = ['EVAL_END', f'Duration: {total_duration:.3f} s']

    for metric in eval_metrics:
        key, value, unit = metric._final()
        key = key[0].upper() + key[1:]
        baselog.append(f'{key}: {value} {unit}')

    logger.info(' | '.join(baselog))
