import logging
import typing as T

import pytorch_lightning as pl
import torch
import transformers
from overrides import overrides
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup
from transformers.modeling_outputs import MaskedLMOutput

SchedulerDefinition = T.Dict[str, T.Union[LambdaLR, str]]

logger = logging.getLogger(__name__)


class RobertaForMaskedLMModule(pl.LightningModule):
    def __init__(
            self,
            job_dir: str,
            max_steps: int,
            learning_rate: float,
            vocab_size: int,
            warmup_steps: int = 100,
            weight_decay: float = 0.01,
    ):
        super().__init__()

        self.job_dir = job_dir
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate

        # Using default RobertaConfig (which uses BertConfig underneath)
        roberta_config = transformers.RobertaConfig(
            # vocab related config
            pad_token_id=3,
            vocab_size=vocab_size,
            # model architecture config
            hidden_size=768,
            num_hidden_layers=4,
            num_attention_heads=12,
        )

        self.model = transformers.AutoModelForMaskedLM.from_config(roberta_config)

    def get_model(self) -> transformers.PreTrainedModel:
        return self.model

    @overrides
    def forward(self, batch, *args, **kwargs) -> MaskedLMOutput:
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

    @overrides
    def training_step(self, batch, batch_idx, *args, **kwargs) -> T.Dict[str, torch.Tensor]:
        mlm_output = self.forward(batch)
        loss = mlm_output.loss
        mlm_accuracy = self.compute_mlm_accuracy(
            batch["labels"],
            mlm_output.logits
        )
        metrics = {
            "loss": loss,
            "mlm_accuracy": mlm_accuracy
        }
        self.log_dict(metrics)
        logger.info(f"step:{self.global_step} - loss: {loss} - train-acc: {mlm_accuracy}")
        return metrics

    @overrides
    def validation_step(self, batch, batch_idx, *args, **kwargs) -> T.Dict[str, torch.Tensor]:
        mlm_output = self.forward(batch)
        mlm_accuracy = self.compute_mlm_accuracy(batch["labels"], mlm_output.logits)
        metrics = {
            "loss": mlm_output.loss,
            "mlm_accuracy": mlm_accuracy
        }
        return metrics

    @overrides
    def validation_epoch_end(self, outputs: T.List[T.Union[torch.Tensor, T.Dict[str, T.Any]]]) -> None:
        loss = torch.mean(torch.FloatTensor([o["loss"] for o in outputs]))
        accuracy = torch.mean(torch.FloatTensor([o["mlm_accuracy"] for o in outputs]))
        perplexity = torch.exp(loss)
        metrics = {
            "val_loss": loss,
            "val_mlm_perplexity": perplexity,
            "val_mlm_accuracy": accuracy
        }
        self.log_dict(metrics)
        logger.info(
            "Metrics for global step {}: {}".format(
                self.global_step,
                {metric_name: "{:.3f}".format(metric_value) for metric_name, metric_value in metrics.items()}
            )
        )

    @staticmethod
    def compute_mlm_accuracy(
            masked_labels: torch.Tensor,
            logits: torch.Tensor
    ) -> torch.Tensor:
        skipping_token_id = -100
        masked_tokens_bool_mask = masked_labels != skipping_token_id

        predictions = logits[masked_tokens_bool_mask, :].argmax(dim=1)
        labels = masked_labels[masked_tokens_bool_mask]
        mlm_accuracy = torch.div(torch.sum(predictions == labels), len(labels))
        return mlm_accuracy

    @overrides
    def configure_optimizers(
            self,
    ) -> T.Tuple[T.List[Optimizer], T.List[SchedulerDefinition]]:
        optimizer = self._get_optimizer()
        scheduler_definition = self._get_scheduler(optimizer)
        return [optimizer], [scheduler_definition]

    def _get_scheduler(self, optimizer) -> T.Dict[str, T.Union[LambdaLR, str]]:
        total_steps = self.max_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, self.warmup_steps, total_steps)
        return {
            "scheduler": scheduler,
            "interval": "step",
        }

    def _get_optimizer(self) -> Optimizer:
        def parameter_applicable_to_weight_decay(parameter_name):
            return not any(no_decay_parameter in parameter_name for no_decay_parameter in ["bias", "LayerNorm.weight"])

        optimizer_grouped_parameters = [
            {
                "params": [
                    parameter for name, parameter in self.model.named_parameters()
                    if parameter_applicable_to_weight_decay(name)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    parameter for name, parameter in self.model.named_parameters()
                    if not parameter_applicable_to_weight_decay(name)
                ],
                "weight_decay": 0.0,
            },
        ]

        return AdamW(
            optimizer_grouped_parameters, lr=self.learning_rate, eps=1e-6
        )
