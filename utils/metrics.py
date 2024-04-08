from typing import final, Union
import torch
import numpy as np
from abc import abstractmethod
from namespace import ExplicitEnum, MetricNames
from utils.processor_utils import EvalAIAnswerProcessor


class ValueNames(ExplicitEnum):
    r"""
    Stores name for input values
    """
    loss = 'loss'
    input_and_label = 'input_and_label'
    resAnswer_and_gtAnswers = 'resAnswer_and_gtAnswers'


def isinstancename(obj, name):
    return obj.__name__ == name


def maybe_evaluate_metric(metrics, value_name: str, value: Union[torch.Tensor, tuple]):
    for metric_object in metrics:
        if value_name not in ValueNames.__dict__.keys():
            member_map = ValueNames.__dict__.keys()
            raise ValueError(f"{value_name} is not a valid metric name, \
                    please select one of {member_map}")

        if any([isinstancename(metric_object, x) for x in [MetricNames.Perplexity, MetricNames.MeanLoss]]) \
            and value_name == ValueNames.loss:
            metric_object(value)
        elif any([isinstancename(metric_object, x) for x in [MetricNames.Accuracy]]) \
            and value_name == ValueNames.input_and_label:
            if not isinstance(value, tuple):
                raise TypeError(f"{value_name} pair must be tuple type")
            metric_object(*value)
        elif isinstancename(metric_object, MetricNames.VQAAccuracy) \
            and value_name == ValueNames.resAnswer_and_gtAnswers:
            if not isinstance(value, tuple):
                raise TypeError(f"{value_name} pair must be tuple type")
            metric_object(*value)


class BaseMetrics(object):

    def __init__(self):
        self.buffer = []

    @abstractmethod
    def __call__(self):
        pass

    @final
    @classmethod
    def final_fn(cls, value, unit):
        if not isinstance(value, float):
            raise ValueError(f"{type(value)} is not a proper type for logging")

        return (cls.__name__, value, unit)


class Perplexity(BaseMetrics):

    def __init__(self):
        super().__init__()
        self.__name__ = MetricNames.Perplexity

    @final
    def __call__(self, loss) -> None:
        self.buffer.append(loss)

    def _final(self) -> tuple:
        self.buffer = [x.item() for x in self.buffer]
        return self.final_fn(np.exp(np.mean(self.buffer)), '')


class Accuracy(BaseMetrics):

    def __init__(self):
        super().__init__()
        self.__name__ = MetricNames.Accuracy

    @final
    def __call__(self, labels, logits, tokenizer) -> None:
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is not None:
            is_not_padded = (labels != pad_token_id)
        else:
            is_not_padded = None
        predict = logits.argmax(dim=-1)
        # self.buffer.append(torch.logical_and(predict == labels, is_not_padded))
        self.buffer.append((predict, labels, is_not_padded))

    def _final(self) -> tuple:
        self.buffer = [torch.logical_and(p == l, pad).mean().item() for p, l, pad in self.buffer]
        return self.final_fn(np.mean(self.buffer), '%')


class MeanLoss(BaseMetrics):

    def __init__(self):
        super().__init__()
        self.__name__ = MetricNames.MeanLoss

    @final
    def __call__(self, loss) -> None:
        self.buffer.append(loss)

    def _final(self) -> tuple:
        self.buffer = [x.item() for x in self.buffer]
        return self.final_fn(np.mean(self.buffer), '')


class VQAAccuracy(BaseMetrics):

    def __init__(self):
        super().__init__()
        self.__name__ = MetricNames.VQAAccuracy
        self.answer_processor = EvalAIAnswerProcessor()

    def _compute_answer_scores(self, raw_answers):
        r"""
        compute the accuracy (soft score) of human answers
        """
        answers = [self.answer_processor(a) for a in raw_answers]
        assert len(answers) == 10
        gt_answers = list(enumerate(answers))
        unique_answers = set(answers)
        unique_answer_scores = {}

        for unique_answer in unique_answers:
            accs = []
            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]
                matching_answers = [item for item in other_answers if item[1] == unique_answer]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)
            unique_answer_scores[unique_answer] = sum(accs) / len(accs)

        return unique_answer_scores

    @final
    def __call__(self, result_answer, gt_answers):
        for data_idx in range(len(result_answer)):
            pred_answer = self.answer_processor(result_answer[data_idx])
            unique_answer_scores = self._compute_answer_scores(gt_answers[data_idx])
            score = unique_answer_scores.get(pred_answer, 0.0)
            self.buffer.append(score)

    def _final(self) -> tuple:
        return self.final_fn(np.mean(self.buffer) * 100, '%')
